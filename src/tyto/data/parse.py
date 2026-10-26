"""OPUS XML + alignment parser for the v2024 OpenSubtitles layout.

v2024 layout (real, confirmed):
  Alignment:  data/raw/opus/v2024/en-ta.xml.gz   (gzip, NOT a zip)
  Tamil docs: data/raw/opus/v2024/ta/<film>.xml.gz  (extracted from ta.zip)
  English docs: data/raw/opus/v2024/en/<film>.xml.gz (fetched via Range from en.zip)

The alignment XML uses colon-separated link types (e.g. ``1:1``, ``1:2``) and
space-separated IDs on each side of the semicolon:

    <linkGrp fromDoc="en/OpenSubtitles.en.ta.12345.xml.gz"
             toDoc="ta/OpenSubtitles.en.ta.12345.xml.gz">
      <link id="SL1" xtargets="1;1" type="1:1"/>
      <link id="SL2" xtargets="1 2;3" type="2:1"/>
    </linkGrp>

The function ``parse_opus_v2024`` is the primary entry point.  For backward
compatibility with tests the old ``parse_opus_zip`` is kept and delegates to
the new code when the path is a directory (cache root) or raises a clear error.

Output schema (pairs.parquet) — unchanged:
    pair_id, film_id, film_title, year, sent_idx,
    en_raw, ta_raw, en, ta, time_start, time_end,
    align_type, source_version, qc_flags

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import zipfile
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from tyto.utils.config import TytoConfig

logger = logging.getLogger(__name__)

# ─── Parquet schema ────────────────────────────────────────────────────────────
PAIRS_SCHEMA = pa.schema(
    [
        pa.field("pair_id", pa.string()),
        pa.field("film_id", pa.string()),
        pa.field("film_title", pa.string()),
        pa.field("year", pa.int16()),
        pa.field("sent_idx", pa.int32()),
        pa.field("en_raw", pa.string()),
        pa.field("ta_raw", pa.string()),
        pa.field("en", pa.string()),
        pa.field("ta", pa.string()),
        pa.field("time_start", pa.float32()),
        pa.field("time_end", pa.float32()),
        pa.field("align_type", pa.string()),
        pa.field("source_version", pa.string()),
        pa.field("qc_flags", pa.string()),
    ]
)

# ─── Shared helpers ───────────────────────────────────────────────────────────

_TITLE_YEAR_RE = re.compile(r"(.+?)\s*\((\d{4})\)\s*$")
_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _parse_time_to_seconds(t: str | None) -> float | None:
    """Convert SRT-style timestamp HH:MM:SS,mmm to seconds (float) or None."""
    if not t:
        return None
    m = _TIME_RE.search(t)
    if not m:
        return None
    h, mn, s, ms = map(int, m.groups())
    return h * 3600.0 + mn * 60.0 + s + ms / 1000.0


def sentences_from_xml_gz_bytes(
    xml_gz_bytes: bytes,
) -> dict[str, tuple[str, float | None, float | None]]:
    """Parse an OPUS per-language XML gzip and return {sent_id: (text, t_start, t_end)}.

    OPUS sentence XML::

        <s id="1">
          <time id="T1S" value="00:00:05,840" type="S"/>
          <w id="1.1">Hello</w>
          <time id="T1E" value="00:00:07,160" type="E"/>
        </s>

    Args:
        xml_gz_bytes: Raw bytes of a gzip-compressed sentence XML file.

    Returns:
        Dict mapping sentence ID -> (plain text, start_seconds, end_seconds).
    """
    result: dict[str, tuple[str, float | None, float | None]] = {}
    try:
        with gzip.open(io.BytesIO(xml_gz_bytes)) as gz:
            tree = ET.parse(gz)
    except Exception as exc:
        logger.warning("Failed to parse sentence XML: %s", exc)
        return result

    for sent_el in tree.iter("s"):
        sid = sent_el.get("id", "")
        words = [w.text or "" for w in sent_el.findall("w")]
        text = " ".join(words).strip()

        t_start: float | None = None
        t_end: float | None = None
        for time_el in sent_el.findall("time"):
            t_type = time_el.get("type", "")
            val = time_el.get("value")
            ts = _parse_time_to_seconds(val)
            if t_type == "S":
                t_start = ts
            elif t_type == "E":
                t_end = ts

        result[sid] = (text, t_start, t_end)
    return result


# Keep the old name as an alias so existing tests don't break.
_sentences_from_xml_gz = sentences_from_xml_gz_bytes


def _normalise_align_type(raw_type: str) -> str:
    """Convert v2024 colon-separated type (e.g. '1:2') to dash form ('1-2').

    Old releases already used dashes; accept both.

    Args:
        raw_type: Raw ``type`` attribute string from a ``<link>`` element.

    Returns:
        Dash-normalised string such as ``'1-2'``.
    """
    return raw_type.replace(":", "-")


def iter_alignment_links(
    align_bytes: bytes,
    *,
    gzipped: bool = True,
) -> Generator[tuple[str, str, list[str], list[str]], None, None]:
    """Iterate over alignment links in an OPUS alignment file.

    Handles both v2024 (plain .xml.gz) and older (embedded in a zip) formats.

    Yields:
        Tuples of ``(from_doc, to_doc, en_ids, ta_ids)``.

    Args:
        align_bytes: Raw bytes of the alignment file (gzip-compressed or plain XML).
        gzipped:     If True, decompress with gzip before parsing.
    """
    try:
        if gzipped:
            with gzip.open(io.BytesIO(align_bytes)) as gz:
                tree = ET.parse(gz)
        else:
            tree = ET.parse(io.BytesIO(align_bytes))  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning("Failed to parse alignment XML: %s", exc)
        return

    for lg in tree.iter("linkGrp"):
        from_doc = lg.get("fromDoc", "")
        to_doc = lg.get("toDoc", "")
        for link in lg.findall("link"):
            xtargets = link.get("xtargets", ";")
            parts = xtargets.split(";")
            en_part = parts[0].strip() if parts else ""
            ta_part = parts[1].strip() if len(parts) > 1 else ""
            en_ids = en_part.split() if en_part else []
            ta_ids = ta_part.split() if ta_part else []
            yield from_doc, to_doc, en_ids, ta_ids


# Legacy alias used by old tests.
def _iter_alignment_links(
    align_gz_bytes: bytes,
) -> Generator[tuple[str, list[str], list[str]], None, None]:
    """Backward-compat wrapper — yields (from_doc, en_ids, ta_ids)."""
    for from_doc, _to_doc, en_ids, ta_ids in iter_alignment_links(align_gz_bytes):
        yield from_doc, en_ids, ta_ids


def _extract_film_meta(from_doc: str) -> tuple[str, str, int | None]:
    """Extract (film_id, film_title, year) from an OPUS fromDoc path.

    v2024 example: ``en/OpenSubtitles.en.ta.12345678.xml.gz``
    → film_id = ``'12345678'``

    Args:
        from_doc: The ``fromDoc`` path string from the alignment XML.

    Returns:
        ``(film_id, film_title, year)`` — title and year may be empty/None.
    """
    stem = from_doc.replace("\\", "/").split("/")[-1]
    stem = re.sub(r"\.xml\.gz$", "", stem)
    # OPUS numeric-ID style: OpenSubtitles.<lang>.<lang>.<numeric_id>
    m_id = re.match(r"OpenSubtitles\.\w+\.\w+\.(\d+)$", stem)
    if m_id:
        return m_id.group(1), "", None

    m_year = _TITLE_YEAR_RE.match(stem)
    if m_year:
        title = m_year.group(1).strip()
        year = int(m_year.group(2))
        film_id = re.sub(r"\W+", "_", title.lower())
        return film_id, title, year

    return stem, stem, None


# ─── v2024 parser ─────────────────────────────────────────────────────────────


def parse_opus_v2024(
    align_gz_path: Path,
    en_cache_dir: Path,
    ta_cache_dir: Path,
    version: str,
    src_lang: str = "en",
    tgt_lang: str = "ta",
    *,
    max_docs: int | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Parse the v2024 OPUS layout into a DataFrame conforming to PAIRS_SCHEMA.

    Args:
        align_gz_path: Local path to the gzip-compressed alignment file
                       (e.g. ``data/raw/opus/v2024/en-ta.xml.gz``).
        en_cache_dir:  Directory containing extracted English XML.gz files.
        ta_cache_dir:  Directory containing extracted Tamil XML.gz files.
        version:       OPUS release version string (stored in ``source_version``).
        src_lang:      Source language code.
        tgt_lang:      Target language code.
        max_docs:      Stop after processing this many documents.
        max_pairs:     Stop after emitting this many alignment pairs.

    Returns:
        DataFrame conforming to PAIRS_SCHEMA.
    """
    from tyto.data.clean import clean_text

    logger.info("Parsing v2024 alignment: %s", align_gz_path)

    # Read the alignment file and group links by document.
    align_bytes = align_gz_path.read_bytes()
    links_by_doc: dict[str, list[tuple[str, list[str], list[str]]]] = defaultdict(list)
    for from_doc, to_doc, en_ids, ta_ids in iter_alignment_links(align_bytes, gzipped=True):
        links_by_doc[from_doc].append((to_doc, en_ids, ta_ids))

    logger.info("Alignment covers %d documents", len(links_by_doc))

    rows: list[dict] = []
    pair_counter = 0
    doc_counter = 0

    for from_doc, link_triples in links_by_doc.items():
        if max_docs is not None and doc_counter >= max_docs:
            break

        film_id, film_title, year = _extract_film_meta(from_doc)

        # Locate the English XML.gz in the cache dir.
        en_fname = from_doc.replace("\\", "/").split("/")[-1]
        en_local = en_cache_dir / en_fname
        if not en_local.exists():
            logger.debug("Skipping %s – English doc not in cache", en_fname)
            continue

        # Locate the Tamil XML.gz; it has the same basename.
        ta_local = ta_cache_dir / en_fname
        if not ta_local.exists():
            logger.debug("Skipping %s – Tamil doc not in cache", en_fname)
            continue

        en_sents = sentences_from_xml_gz_bytes(en_local.read_bytes())
        ta_sents = sentences_from_xml_gz_bytes(ta_local.read_bytes())

        for sent_idx, (_to_doc, en_ids, ta_ids) in enumerate(link_triples, start=1):
            en_texts = [en_sents.get(sid, ("", None, None))[0] for sid in en_ids]
            ta_texts = [ta_sents.get(sid, ("", None, None))[0] for sid in ta_ids]

            en_raw = " ".join(t for t in en_texts if t).strip()
            ta_raw = " ".join(t for t in ta_texts if t).strip()

            if not en_raw and not ta_raw:
                continue

            # Timing from English sentences.
            t_start: float | None = None
            t_end: float | None = None
            for sid in en_ids:
                entry = en_sents.get(sid)
                if entry:
                    if t_start is None and entry[1] is not None:
                        t_start = entry[1]
                    if entry[2] is not None:
                        t_end = entry[2]

            # v2024 stores type as "1:1"; normalise to "1-1".
            raw_type = f"{len(en_ids)}:{len(ta_ids)}"
            align_type = _normalise_align_type(raw_type)

            en_clean = clean_text(en_raw, lang=src_lang)
            ta_clean = clean_text(ta_raw, lang=tgt_lang)

            rows.append(
                {
                    "pair_id": f"{film_id}:{sent_idx}",
                    "film_id": film_id,
                    "film_title": film_title or "",
                    "year": year,
                    "sent_idx": sent_idx,
                    "en_raw": en_raw,
                    "ta_raw": ta_raw,
                    "en": en_clean,
                    "ta": ta_clean,
                    "time_start": t_start,
                    "time_end": t_end,
                    "align_type": align_type,
                    "source_version": version,
                    "qc_flags": "",
                }
            )
            pair_counter += 1
            if max_pairs is not None and pair_counter >= max_pairs:
                break

        doc_counter += 1
        if max_pairs is not None and pair_counter >= max_pairs:
            logger.info("Reached max_pairs=%d, stopping early.", max_pairs)
            break
        if doc_counter % 200 == 0:
            logger.info("Parsed %d docs, %d pairs …", doc_counter, pair_counter)

    logger.info(
        "Parsing complete: %d pairs from %d documents", pair_counter, doc_counter
    )
    return _rows_to_df(rows)


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of row dicts to a typed DataFrame matching PAIRS_SCHEMA."""
    if not rows:
        return pd.DataFrame({col: pd.Series([], dtype=str) for col in PAIRS_SCHEMA.names})
    df = pd.DataFrame(rows)
    df["year"] = pd.array(df["year"], dtype=pd.Int16Dtype())
    df["sent_idx"] = df["sent_idx"].astype("int32")
    df["time_start"] = df["time_start"].astype("float32")
    df["time_end"] = df["time_end"].astype("float32")
    return df


# ─── Legacy zip parser (kept for backward compatibility with tests) ────────────


def parse_opus_zip(
    zip_path: Path,
    version: str,
    src_lang: str = "en",
    tgt_lang: str = "ta",
    *,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Parse an OPUS XML zip archive (pre-v2024 layout) into a DataFrame.

    This function handles the old layout where a single ``en-ta.zip`` contained
    all sentence files plus an embedded alignment file.  It is kept for test
    backward compatibility.

    Args:
        zip_path:  Path to the OPUS XML zip.
        version:   OPUS release version string.
        src_lang:  Source language code.
        tgt_lang:  Target language code.
        max_pairs: If set, stop after this many pairs.

    Returns:
        DataFrame conforming to PAIRS_SCHEMA.
    """
    from tyto.data.clean import clean_text

    rows: list[dict] = []
    pair_counter = 0

    logger.info("Opening OPUS zip (legacy layout): %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find alignment file.
        align_entries = [
            n
            for n in zf.namelist()
            if n.endswith(f"{src_lang}-{tgt_lang}.xml.gz")
            or n.endswith(f"{tgt_lang}-{src_lang}.xml.gz")
        ]
        if not align_entries:
            raise FileNotFoundError(
                f"No alignment file in {zip_path} for {src_lang}-{tgt_lang}"
            )
        align_gz_bytes = zf.read(align_entries[0])
        logger.info("Using alignment file: %s", align_entries[0])

        links_by_doc: dict[str, list[tuple[list[str], list[str]]]] = defaultdict(list)
        for from_doc, en_ids, ta_ids in _iter_alignment_links(align_gz_bytes):
            links_by_doc[from_doc].append((en_ids, ta_ids))

        for doc_idx, (from_doc, link_pairs) in enumerate(links_by_doc.items()):
            film_id, film_title, year = _extract_film_meta(from_doc)

            # Derive Tamil path from English path.
            to_doc = from_doc.replace(f"/{src_lang}/", f"/{tgt_lang}/", 1)
            to_doc = to_doc.replace(f"{src_lang}.{tgt_lang}", f"{tgt_lang}.{src_lang}", 1)

            en_entry = _resolve_gz_in_zip(zf, src_lang, from_doc)
            ta_entry = _resolve_gz_in_zip(zf, tgt_lang, to_doc)
            if ta_entry is None:
                alt = from_doc.replace(f"/{src_lang}/", f"/{tgt_lang}/", 1)
                ta_entry = _resolve_gz_in_zip(zf, tgt_lang, alt)

            if en_entry is None or ta_entry is None:
                logger.debug("Skipping %s – XML not in zip", from_doc)
                continue

            en_sents = sentences_from_xml_gz_bytes(zf.read(en_entry))
            ta_sents = sentences_from_xml_gz_bytes(zf.read(ta_entry))

            for sent_idx, (en_ids, ta_ids) in enumerate(link_pairs, start=1):
                en_raw = " ".join(
                    en_sents.get(sid, ("", None, None))[0] for sid in en_ids
                ).strip()
                ta_raw = " ".join(
                    ta_sents.get(sid, ("", None, None))[0] for sid in ta_ids
                ).strip()
                if not en_raw and not ta_raw:
                    continue

                t_start: float | None = None
                t_end: float | None = None
                for sid in en_ids:
                    entry = en_sents.get(sid)
                    if entry:
                        if t_start is None and entry[1] is not None:
                            t_start = entry[1]
                        if entry[2] is not None:
                            t_end = entry[2]

                rows.append(
                    {
                        "pair_id": f"{film_id}:{sent_idx}",
                        "film_id": film_id,
                        "film_title": film_title or "",
                        "year": year,
                        "sent_idx": sent_idx,
                        "en_raw": en_raw,
                        "ta_raw": ta_raw,
                        "en": clean_text(en_raw, lang=src_lang),
                        "ta": clean_text(ta_raw, lang=tgt_lang),
                        "time_start": t_start,
                        "time_end": t_end,
                        "align_type": f"{len(en_ids)}-{len(ta_ids)}",
                        "source_version": version,
                        "qc_flags": "",
                    }
                )
                pair_counter += 1
                if max_pairs is not None and pair_counter >= max_pairs:
                    break

            if max_pairs is not None and pair_counter >= max_pairs:
                break
            if doc_idx % 500 == 0:
                logger.info("Parsed %d docs, %d pairs …", doc_idx + 1, pair_counter)

    return _rows_to_df(rows)


def _resolve_gz_in_zip(zf: zipfile.ZipFile, lang: str, doc_path: str) -> str | None:
    """Find a zip entry by suffix + language folder membership."""
    filename = doc_path.replace("\\", "/").split("/")[-1]
    candidates = [
        n for n in zf.namelist() if n.endswith(filename) and f"/{lang}/" in n
    ]
    if candidates:
        return candidates[0]
    candidates = [n for n in zf.namelist() if n.endswith(filename)]
    return candidates[0] if candidates else None


# Keep the old name in case something imported it directly.
_resolve_gz_path_in_zip = _resolve_gz_in_zip


# ─── Top-level save function ──────────────────────────────────────────────────


def parse_and_save(
    cfg: TytoConfig,
    source: Path,
    output_path: Path | None = None,
    *,
    max_docs: int | None = None,
    max_pairs: int | None = None,
) -> Path:
    """Parse OPUS data and write pairs.parquet (idempotent, overwrites).

    Detects the layout automatically:
    - If *source* is a directory (the v2024 cache root), calls
      ``parse_opus_v2024``.
    - If *source* is a .zip file (legacy layout), calls ``parse_opus_zip``.

    Args:
        cfg:         TytoConfig.
        source:      Either the v2024 cache root directory or a legacy zip path.
        output_path: Destination parquet. Defaults to
                     ``cfg.paths.processed_dir / 'pairs.parquet'``.
        max_docs:    Limit on documents (v2024 only).
        max_pairs:   Limit on alignment pairs (both layouts).

    Returns:
        Path to the written parquet file.
    """
    if output_path is None:
        output_path = Path(cfg.paths.processed_dir) / "pairs.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    opus = cfg.opus
    version = opus.version

    if source.is_dir():
        # v2024 cache root layout.
        align_gz = source / f"{opus.src_lang}-{opus.tgt_lang}.xml.gz"
        en_cache = source / opus.src_lang
        ta_cache = source / opus.tgt_lang
        if not align_gz.exists():
            raise FileNotFoundError(
                f"Alignment file not found: {align_gz}. "
                "Run 'python -m tyto data download' first."
            )
        df = parse_opus_v2024(
            align_gz,
            en_cache,
            ta_cache,
            version,
            src_lang=opus.src_lang,
            tgt_lang=opus.tgt_lang,
            max_docs=max_docs,
            max_pairs=max_pairs,
        )
    else:
        # Legacy zip layout.
        df = parse_opus_zip(
            source,
            version,
            src_lang=opus.src_lang,
            tgt_lang=opus.tgt_lang,
            max_pairs=max_pairs,
        )

    table = pa.Table.from_pandas(df, schema=PAIRS_SCHEMA, safe=False)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Wrote %d pairs to %s", len(df), output_path)
    return output_path

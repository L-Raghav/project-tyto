"""OPUS XML + alignment parser: converts the en-ta OpenSubtitles zip into pairs.parquet.

The OPUS XML layout for OpenSubtitles::

    <lang>/           - one .xml.gz per film, containing <s id="…"><w>token</w></s> elements
    xml/en-ta.xml.gz  - alignment file containing <linkGrp> / <link> elements

Output schema (pairs.parquet):

    pair_id        str     – globally unique "{film_id}:{sent_idx}"
    film_id        str     – OPUS internal film identifier
    film_title     str     – human-readable title (nullable)
    year           int16   – release year (nullable)
    sent_idx       int32   – 1-based sentence index within the film
    en_raw         str     – raw English sentence (pre-clean)
    ta_raw         str     – raw Tamil sentence (pre-clean)
    en             str     – cleaned English sentence
    ta             str     – cleaned Tamil sentence
    time_start     float32 – subtitle start time in seconds (nullable)
    time_end       float32 – subtitle end time in seconds (nullable)
    align_type     str     – "1-1" | "1-2" | "2-1" | "1-0" | "0-1" | etc.
    source_version str     – OPUS corpus version string (e.g. "2016")
    qc_flags       str     – pipe-separated QC flag names (empty string = clean)

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import zipfile
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

# ─── Helpers ──────────────────────────────────────────────────────────────────

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


def _sentences_from_xml_gz(
    xml_gz_bytes: bytes,
) -> dict[str, tuple[str, float | None, float | None]]:
    """Parse an OPUS per-language XML gzip and return {sent_id: (text, t_start, t_end)}.

    OPUS sentence XML looks like::

        <s id="1">
          <time id="T1S" value="00:00:05,840" type="S"/>
          <w id="1.1">Hello</w>
          <w id="1.2">world</w>
          <time id="T1E" value="00:00:07,160" type="E"/>
        </s>

    Args:
        xml_gz_bytes: Raw bytes of a gzip-compressed OPUS sentence XML file.

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


def _iter_alignment_links(
    align_gz_bytes: bytes,
) -> Generator[tuple[str, list[str], list[str]], None, None]:
    """Iterate over alignment links from an OPUS alignment XML.gz.

    Yields: (doc_id, en_ids, ta_ids)

    OPUS alignment XML looks like::

        <cesAlign version="1.0">
          <linkGrp targType="s" toDoc="ta/film.xml.gz" fromDoc="en/film.xml.gz">
            <link id="SL1" xtargets="1 2;3" type="2-1"/>
          </linkGrp>
        </cesAlign>

    Args:
        align_gz_bytes: Raw bytes of the gzip-compressed alignment file.

    Yields:
        Tuples of (link_group_from_doc, list_of_en_sids, list_of_ta_sids).
    """
    try:
        with gzip.open(io.BytesIO(align_gz_bytes)) as gz:
            tree = ET.parse(gz)
    except Exception as exc:
        logger.warning("Failed to parse alignment XML: %s", exc)
        return

    for lg in tree.iter("linkGrp"):
        from_doc = lg.get("fromDoc", "")
        for link in lg.findall("link"):
            xtargets = link.get("xtargets", ";")
            parts = xtargets.split(";")
            en_part = parts[0].strip() if len(parts) > 0 else ""
            ta_part = parts[1].strip() if len(parts) > 1 else ""
            en_ids = en_part.split() if en_part else []
            ta_ids = ta_part.split() if ta_part else []
            yield (from_doc, en_ids, ta_ids)


def _extract_film_meta(from_doc: str) -> tuple[str, str, int | None]:
    """Extract film_id, title guess, and year from an OPUS fromDoc path.

    Example path: ``en/OpenSubtitles.en.ta.123456789.xml.gz``
    or            ``en/Some Title (2010).xml.gz``

    Returns:
        (film_id, film_title, year)  where film_title / year may be empty/None.
    """
    # Normalise separators
    stem = from_doc.replace("\\", "/").split("/")[-1]
    # Remove .xml.gz
    stem = re.sub(r"\.xml\.gz$", "", stem)
    # OPUS numeric-ID style: OpenSubtitles.<lang>.<lang>.<numeric_id>
    m_id = re.match(r"OpenSubtitles\.\w+\.\w+\.(\d+)$", stem)
    if m_id:
        film_id = m_id.group(1)
        return film_id, "", None

    # Fallback: use full stem as film_id, try to parse "(year)" suffix.
    m_year = _TITLE_YEAR_RE.match(stem)
    if m_year:
        title = m_year.group(1).strip()
        year = int(m_year.group(2))
        film_id = re.sub(r"\W+", "_", title.lower())
        return film_id, title, year

    return stem, stem, None


def _resolve_gz_path_in_zip(
    zf: zipfile.ZipFile, lang: str, doc_path: str
) -> str | None:
    """Find the actual zip entry for an OPUS per-language document.

    OPUS packs files under ``<corpus>/<version>/xml/<lang>/<film>.xml.gz`` or
    just ``<lang>/<film>.xml.gz``.  We search by suffix match.

    Args:
        zf:       Open ZipFile.
        lang:     Language code, e.g. ``"en"`` or ``"ta"``.
        doc_path: The ``fromDoc`` / ``toDoc`` path from the alignment XML.

    Returns:
        The matching zip entry name, or None if not found.
    """
    # doc_path is e.g. "en/OpenSubtitles.en.ta.12345.xml.gz"
    # Normalise to just the filename part after the language folder.
    filename = doc_path.replace("\\", "/").split("/")[-1]
    candidates = [
        name for name in zf.namelist()
        if name.endswith(filename) and f"/{lang}/" in name
    ]
    if candidates:
        return candidates[0]
    # Looser search: any entry ending with the filename under a "lang" component.
    candidates = [
        name for name in zf.namelist()
        if name.endswith(filename)
    ]
    return candidates[0] if candidates else None


def parse_opus_zip(
    zip_path: Path,
    version: str,
    src_lang: str = "en",
    tgt_lang: str = "ta",
    *,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Parse an OPUS XML zip archive into a DataFrame with the pairs schema.

    The zip contains:

    - ``<lang>/<film>.xml.gz`` – per-language sentence files
    - ``xml/<src>-<tgt>.xml.gz`` – alignment file at the root of the xml/ folder

    Args:
        zip_path:  Path to the downloaded OPUS XML zip.
        version:   OPUS release version string (stored in ``source_version``).
        src_lang:  Source language code (``"en"``).
        tgt_lang:  Target language code (``"ta"``).
        max_pairs: If set, stop after this many pairs (useful for smoke tests).

    Returns:
        DataFrame conforming to ``PAIRS_SCHEMA``.
    """
    from tyto.data.clean import clean_text

    rows: list[dict] = []
    pair_counter = 0

    logger.info("Opening OPUS zip: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the alignment file.
        align_entries = [
            n for n in zf.namelist()
            if n.endswith(f"{src_lang}-{tgt_lang}.xml.gz") or
               n.endswith(f"{tgt_lang}-{src_lang}.xml.gz")
        ]
        if not align_entries:
            raise FileNotFoundError(
                f"No alignment file found in {zip_path} "
                f"for pair {src_lang}-{tgt_lang}"
            )
        align_entry = align_entries[0]
        logger.info("Using alignment file: %s", align_entry)

        align_gz_bytes = zf.read(align_entry)

        # Group links by from_doc so we load each sentence XML only once.
        from collections import defaultdict

        links_by_doc: dict[str, list[tuple[list[str], list[str]]]] = defaultdict(list)
        for from_doc, en_ids, ta_ids in _iter_alignment_links(align_gz_bytes):
            links_by_doc[from_doc].append((en_ids, ta_ids))

        logger.info("Found %d documents in alignment file", len(links_by_doc))

        for doc_idx, (from_doc, link_pairs) in enumerate(links_by_doc.items()):
            film_id, film_title, year = _extract_film_meta(from_doc)

            # Derive the ta doc path: replace the lang prefix.
            to_doc = from_doc.replace(f"/{src_lang}/", f"/{tgt_lang}/", 1)
            to_doc = to_doc.replace(
                f"{src_lang}.{tgt_lang}",
                f"{tgt_lang}.{src_lang}",
                1,
            )
            # Also try the reverse lang order in the filename.
            to_doc_alt = from_doc.replace(
                f"/{src_lang}/",
                f"/{tgt_lang}/",
                1,
            )

            en_zip_path = _resolve_gz_path_in_zip(zf, src_lang, from_doc)
            ta_zip_path = _resolve_gz_path_in_zip(zf, tgt_lang, to_doc)
            if ta_zip_path is None:
                ta_zip_path = _resolve_gz_path_in_zip(zf, tgt_lang, to_doc_alt)

            if en_zip_path is None or ta_zip_path is None:
                logger.debug(
                    "Skipping doc %s – could not locate en or ta XML in zip", from_doc
                )
                continue

            en_sents = _sentences_from_xml_gz(zf.read(en_zip_path))
            ta_sents = _sentences_from_xml_gz(zf.read(ta_zip_path))

            for sent_idx, (en_ids, ta_ids) in enumerate(link_pairs, start=1):
                # Build merged text for each side.
                en_texts = [en_sents.get(sid, ("", None, None))[0] for sid in en_ids]
                ta_texts = [ta_sents.get(sid, ("", None, None))[0] for sid in ta_ids]

                en_raw = " ".join(t for t in en_texts if t).strip()
                ta_raw = " ".join(t for t in ta_texts if t).strip()

                # Skip 0-0 alignments.
                if not en_raw and not ta_raw:
                    continue

                # Timing: use first en sentence's start and last en sentence's end.
                t_start: float | None = None
                t_end: float | None = None
                for sid in en_ids:
                    entry = en_sents.get(sid)
                    if entry:
                        if t_start is None and entry[1] is not None:
                            t_start = entry[1]
                        if entry[2] is not None:
                            t_end = entry[2]

                align_type = f"{len(en_ids)}-{len(ta_ids)}"
                pair_id = f"{film_id}:{sent_idx}"

                en_clean = clean_text(en_raw, lang="en")
                ta_clean = clean_text(ta_raw, lang="ta")

                rows.append(
                    {
                        "pair_id": pair_id,
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

            if max_pairs is not None and pair_counter >= max_pairs:
                logger.info("Reached max_pairs=%d limit, stopping early.", max_pairs)
                break

            if doc_idx % 500 == 0:
                logger.info(
                    "Parsed %d documents, %d pairs so far …", doc_idx + 1, pair_counter
                )

    logger.info("Parsing complete: %d pairs from %d documents", pair_counter, len(links_by_doc))

    if not rows:
        return pd.DataFrame(
            {col: pd.Series([], dtype=str) for col in PAIRS_SCHEMA.names}
        )

    df = pd.DataFrame(rows)
    # Cast typed columns.
    df["year"] = pd.array(df["year"], dtype=pd.Int16Dtype())
    df["sent_idx"] = df["sent_idx"].astype("int32")
    df["time_start"] = df["time_start"].astype("float32")
    df["time_end"] = df["time_end"].astype("float32")
    return df


def parse_and_save(
    cfg: TytoConfig,
    zip_path: Path,
    output_path: Path | None = None,
    *,
    max_pairs: int | None = None,
) -> Path:
    """Parse OPUS zip and write pairs.parquet, idempotent (overwrites if exists).

    Args:
        cfg:         TytoConfig (for version, lang settings).
        zip_path:    Path to the downloaded XML zip.
        output_path: Destination parquet file. Defaults to
                     ``cfg.paths.processed_dir / "pairs.parquet"``.
        max_pairs:   Optional limit for smoke tests.

    Returns:
        Path to the written parquet file.
    """
    if output_path is None:
        output_path = Path(cfg.paths.processed_dir) / "pairs.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = parse_opus_zip(
        zip_path,
        version=cfg.opus.version,
        src_lang=cfg.opus.src_lang,
        tgt_lang=cfg.opus.tgt_lang,
        max_pairs=max_pairs,
    )

    table = pa.Table.from_pandas(df, schema=PAIRS_SCHEMA, safe=False)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("Wrote %d pairs to %s", len(df), output_path)
    return output_path

"""v2018 OPUS OpenSubtitles en-ta loader.

Reads the zip directly (no extraction), parses .en / .ta / .ids files,
applies NFC + whitespace cleaning, runs per-document integrity checks, and
writes data/processed/pairs_v2018.parquet.
"""

from __future__ import annotations

import logging
import unicodedata
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Vowel signs in Tamil Unicode block (U+0BBE – U+0BCC)
_VOWEL_SIGN_RANGE = (0x0BBE, 0x0BCC)
# IMDb IDs that belong to series_A (hypothesis — to be verified)
_SERIES_A_IDS: frozenset[str] = frozenset(
    ["6261062", "6261064", "6261068", "6261070", "6261074", "6261076"]
)
_EXPECTED_ROWS = 32_417
_VOWEL_GATE = 0.10


def _nfc_clean(text: str) -> str:
    """NFC-normalise and strip leading/trailing whitespace."""
    return unicodedata.normalize("NFC", text).strip()


def _vowel_share(ta_text: str) -> float:
    """Return fraction of Tamil chars that are vowel signs (U+0BBE-U+0BCC)."""
    lo, hi = _VOWEL_SIGN_RANGE
    total = len(ta_text)
    if total == 0:
        return 0.0
    vowels = sum(1 for ch in ta_text if lo <= ord(ch) <= hi)
    return vowels / total


def _link_type(en_sids: str, ta_sids: str) -> str:
    """Return '1:1', '1:2', '2:1', etc. based on sentence ID counts."""
    n_en = len(en_sids.split()) if en_sids.strip() else 0
    n_ta = len(ta_sids.split()) if ta_sids.strip() else 0
    return f"{n_en}:{n_ta}"


def _film_id_from_path(path: str) -> str:
    """Extract IMDb film ID (3rd path component) from an OPUS path."""
    parts = path.strip("/").split("/")
    return parts[2] if len(parts) > 2 else ""


def _year_from_path(path: str) -> str:
    """Extract year (2nd path component) from an OPUS path."""
    parts = path.strip("/").split("/")
    return parts[1] if len(parts) > 1 else ""


def load_v2018(
    zip_path: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:
    """Load the v2018 OPUS en-ta corpus from the zip and write a parquet file.

    Parameters
    ----------
    zip_path:
        Path to ``en-ta.txt.zip`` (the v2018 OPUS release).
    out_path:
        Destination path for ``pairs_v2018.parquet``.

    Returns
    -------
    pd.DataFrame
        The parsed and validated DataFrame (32 417 rows).

    Raises
    ------
    RuntimeError
        If any document's Tamil vowel-sign share is below 0.10, or the
        total row count is not 32 417.
    """
    zip_path = Path(zip_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Opening zip: %s", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        logger.debug("Zip members: %s", names)

        # Locate the three member files
        en_name = next(n for n in names if n.endswith(".en"))
        ta_name = next(n for n in names if n.endswith(".ta"))
        ids_name = next(n for n in names if n.endswith(".ids"))

        logger.info("Reading %s", en_name)
        en_lines = zf.read(en_name).decode("utf-8").splitlines()
        logger.info("Reading %s", ta_name)
        ta_lines = zf.read(ta_name).decode("utf-8").splitlines()
        logger.info("Reading %s", ids_name)
        ids_lines = zf.read(ids_name).decode("utf-8").splitlines()

    if not (len(en_lines) == len(ta_lines) == len(ids_lines)):
        raise RuntimeError(
            f"File length mismatch: en={len(en_lines)}, "
            f"ta={len(ta_lines)}, ids={len(ids_lines)}"
        )

    logger.info("Parsing %d lines", len(en_lines))

    rows: list[dict[str, Any]] = []
    for i, (en_raw, ta_raw, ids_line) in enumerate(
        zip(en_lines, ta_lines, ids_lines, strict=True)
    ):
        ids_parts = ids_line.split("\t")
        if len(ids_parts) != 4:
            logger.warning("Line %d: unexpected ids columns (%d)", i, len(ids_parts))
            en_path, ta_path, en_sids, ta_sids = (
                ids_parts + ["", "", "", ""]
            )[:4]
        else:
            en_path, ta_path, en_sids, ta_sids = ids_parts

        film_id = _film_id_from_path(en_path)
        year = _year_from_path(en_path)
        en_doc = en_path
        ta_doc = ta_path
        series_group = "series_A" if film_id in _SERIES_A_IDS else ""

        en_clean = _nfc_clean(en_raw)
        ta_clean = _nfc_clean(ta_raw)
        link_type = _link_type(en_sids, ta_sids)

        rows.append(
            {
                "pair_id": i,
                "film_id": film_id,
                "series_group": series_group,
                "year": year,
                "en_doc": en_doc,
                "ta_doc": ta_doc,
                "en_sids": en_sids,
                "ta_sids": ta_sids,
                "link_type": link_type,
                "en": en_clean,
                "ta": ta_clean,
                "en_raw": en_raw,
                "ta_raw": ta_raw,
            }
        )

    df = pd.DataFrame(rows)

    # --- Stop gate: total row count ---
    if len(df) != _EXPECTED_ROWS:
        raise RuntimeError(
            f"Row count mismatch: got {len(df)}, expected {_EXPECTED_ROWS}"
        )

    # --- Per-document integrity check: vowel-sign share ---
    logger.info("Running per-document vowel-share integrity check …")
    doc_groups = df.groupby("en_doc")
    any_failed = False
    for doc_name, group in doc_groups:
        ta_all = "".join(group["ta"].tolist())
        vs = _vowel_share(ta_all)
        logger.info("  doc=%s  vowel_share=%.4f", doc_name, vs)
        print(f"  vowel_share={vs:.4f}  doc={doc_name}")
        if vs < _VOWEL_GATE:
            logger.error(
                "INTEGRITY FAIL: doc %s has vowel_share=%.4f < %.2f",
                doc_name,
                vs,
                _VOWEL_GATE,
            )
            any_failed = True
    if any_failed:
        raise RuntimeError(
            "One or more documents failed the Tamil vowel-sign gate (<0.10). "
            "Check logger output for details."
        )

    # --- Write parquet ---
    logger.info("Writing parquet to %s", out_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, str(out_path))

    n_docs = df["en_doc"].nunique()
    print(
        f"\nload_v2018 complete: {len(df):,} pairs, {n_docs} distinct documents → {out_path}"
    )
    logger.info("Done: %d pairs, %d documents", len(df), n_docs)
    return df

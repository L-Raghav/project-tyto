"""SRT temporal-overlap aligner: aligns two .srt files by Intersection-over-Union (IoU).

Given English and Tamil .srt files from the same film, this module:

1. Parses both SRT files with the ``srt`` library.
2. Computes temporal IoU between every en/ta subtitle segment pair.
3. Greedily matches pairs with IoU ≥ ``min_iou`` (configurable).
4. Merges split lines (multiple SRT entries matched to one on the other side).
5. Returns a DataFrame conforming to the ``PAIRS_SCHEMA`` from ``parse.py``.

Usage::

    from tyto.data.srt_align import align_srt_files
    df = align_srt_files("movie.en.srt", "movie.ta.srt")

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import srt

if TYPE_CHECKING:
    from datetime import timedelta

logger = logging.getLogger(__name__)


def _td_to_seconds(td: timedelta) -> float:
    """Convert a ``datetime.timedelta`` to seconds (float)."""
    return td.total_seconds()


def _iou(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    """Compute temporal Intersection-over-Union for two intervals.

    Args:
        start_a, end_a: Interval A in seconds.
        start_b, end_b: Interval B in seconds.

    Returns:
        IoU in [0, 1].  Returns 0.0 for zero-length intervals.
    """
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0.0, inter_end - inter_start)
    if intersection == 0.0:
        return 0.0
    union = (end_a - start_a) + (end_b - start_b) - intersection
    return intersection / union if union > 0.0 else 0.0


def _merge_srt_content(subtitles: list[srt.Subtitle]) -> str:
    """Concatenate the content of multiple SRT subtitles, stripping markup."""
    from tyto.data.clean import clean_text

    texts = [clean_text(sub.content, lang="en") for sub in subtitles]
    return " ".join(t for t in texts if t).strip()


def align_srt_files(
    en_path: str | Path,
    ta_path: str | Path,
    *,
    film_id: str = "unknown",
    film_title: str = "",
    year: int | None = None,
    source_version: str = "srt",
    min_iou: float = 0.1,
) -> pd.DataFrame:
    """Align two SRT subtitle files by temporal overlap (IoU).

    The algorithm:
    1. Parse both SRT files.
    2. For each Tamil subtitle, find all English subtitles with IoU ≥ min_iou.
    3. Merge any split English lines that together match a single Tamil subtitle,
       and vice-versa.
    4. Emit one pair per matched group.

    Pairs are returned in temporal order with ``sent_idx`` starting from 1.

    Args:
        en_path:        Path to the English SRT file.
        ta_path:        Path to the Tamil SRT file.
        film_id:        Identifier for the film (used in ``pair_id``).
        film_title:     Human-readable film title (nullable).
        year:           Release year (nullable).
        source_version: Stored in ``source_version`` column.
        min_iou:        Minimum IoU threshold to consider a pair aligned.

    Returns:
        DataFrame conforming to ``PAIRS_SCHEMA`` from ``tyto.data.parse``.
    """
    from tyto.data.clean import clean_text, compute_script_flags
    from tyto.data.parse import PAIRS_SCHEMA

    en_path = Path(en_path)
    ta_path = Path(ta_path)

    with en_path.open(encoding="utf-8", errors="replace") as fh:
        en_subs = list(srt.parse(fh.read()))
    with ta_path.open(encoding="utf-8", errors="replace") as fh:
        ta_subs = list(srt.parse(fh.read()))

    logger.info(
        "Aligning %s: %d EN / %d TA subtitles", film_id, len(en_subs), len(ta_subs)
    )

    # Build interval arrays for fast overlap search.
    en_intervals = [
        (_td_to_seconds(s.start), _td_to_seconds(s.end), s)
        for s in en_subs
    ]
    ta_intervals = [
        (_td_to_seconds(s.start), _td_to_seconds(s.end), s)
        for s in ta_subs
    ]

    # For each Tamil subtitle, collect all overlapping English subtitles.
    rows: list[dict] = []
    used_en: set[int] = set()  # track merged EN indices

    for ta_idx, (ta_start, ta_end, ta_sub) in enumerate(ta_intervals):
        overlapping_en: list[tuple[float, int, srt.Subtitle]] = []

        for en_idx, (en_start, en_end, en_sub) in enumerate(en_intervals):
            if en_end < ta_start - 2.0:
                continue  # too early, skip
            if en_start > ta_end + 2.0:
                break  # too late, stop

            iou = _iou(ta_start, ta_end, en_start, en_end)
            if iou >= min_iou:
                overlapping_en.append((iou, en_idx, en_sub))

        if not overlapping_en:
            continue

        # Sort by IoU desc and take all overlapping matches.
        overlapping_en.sort(key=lambda x: -x[0])
        matched_en_subs = [sub for _, _, sub in overlapping_en]
        matched_en_indices = [idx for _, idx, _ in overlapping_en]

        # Mark EN indices as used.
        used_en.update(matched_en_indices)

        # Build timing from the full matched window.
        t_start = min(en_intervals[i][0] for i in matched_en_indices)
        t_end = max(en_intervals[i][1] for i in matched_en_indices)

        en_raw = " ".join(
            sub.content for sub in sorted(matched_en_subs, key=lambda s: s.start)
        ).strip()
        ta_raw = ta_sub.content.strip()

        en_clean = _merge_srt_content(
            sorted(matched_en_subs, key=lambda s: s.start)
        )
        ta_clean = clean_text(ta_raw, lang="ta")

        align_type = f"{len(matched_en_subs)}-1"
        sent_idx = ta_idx + 1
        pair_id = f"{film_id}:{sent_idx}"

        # Compute QC flags.
        flags: list[str] = []
        flags.extend(compute_script_flags(ta_clean))

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
                "time_start": float(t_start),
                "time_end": float(t_end),
                "align_type": align_type,
                "source_version": source_version,
                "qc_flags": "|".join(flags),
            }
        )

    logger.info("SRT align produced %d aligned pairs for %s", len(rows), film_id)

    if not rows:
        return pd.DataFrame({col: pd.Series([], dtype=str) for col in PAIRS_SCHEMA.names})

    df = pd.DataFrame(rows)
    df["year"] = pd.array(df["year"], dtype=pd.Int16Dtype())
    df["sent_idx"] = df["sent_idx"].astype("int32")
    df["time_start"] = df["time_start"].astype("float32")
    df["time_end"] = df["time_end"].astype("float32")
    return df


def align_srt_pair_to_parquet(
    en_path: str | Path,
    ta_path: str | Path,
    output_path: str | Path,
    *,
    film_id: str = "unknown",
    film_title: str = "",
    year: int | None = None,
    source_version: str = "srt",
    min_iou: float = 0.1,
) -> Path:
    """Align two SRT files and write the result as a Parquet file.

    Args:
        en_path:        Path to English SRT.
        ta_path:        Path to Tamil SRT.
        output_path:    Destination Parquet file path.
        film_id:        Film identifier.
        film_title:     Human-readable title.
        year:           Release year.
        source_version: Source label.
        min_iou:        IoU threshold.

    Returns:
        Path to the written Parquet file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tyto.data.parse import PAIRS_SCHEMA

    df = align_srt_files(
        en_path,
        ta_path,
        film_id=film_id,
        film_title=film_title,
        year=year,
        source_version=source_version,
        min_iou=min_iou,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, schema=PAIRS_SCHEMA, safe=False)
    pq.write_table(table, output_path, compression="snappy")
    logger.info("SRT pair parquet written to %s (%d rows)", output_path, len(df))
    return output_path

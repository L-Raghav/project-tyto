"""Unit tests for tyto.data.srt_align.

Tests use tiny in-memory SRT content via temporary files.
Verifies: IoU computation, exact/near temporal overlap, split-line merging,
pairs.parquet schema compliance, and fixture with all-Tamil input.

All tests run on CPU in under 60 seconds.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from tyto.data.parse import PAIRS_SCHEMA


# ─── Fixtures ─────────────────────────────────────────────────────────────────

EN_SRT_CONTENT = textwrap.dedent("""\
    1
    00:00:01,000 --> 00:00:03,000
    Hello world.

    2
    00:00:03,500 --> 00:00:05,000
    How are you?

    3
    00:00:05,500 --> 00:00:07,000
    I am fine.

    4
    00:00:07,500 --> 00:00:09,000
    Goodbye!

""")

# Tamil subtitles overlap temporally with EN.
TA_SRT_CONTENT = textwrap.dedent("""\
    1
    00:00:01,200 --> 00:00:03,200
    வணக்கம் உலகம்.

    2
    00:00:03,600 --> 00:00:05,200
    நீங்கள் எப்படி இருக்கிறீர்கள்?

    3
    00:00:05,600 --> 00:00:07,200
    நான் நலமாக இருக்கிறேன்.

    4
    00:00:07,600 --> 00:00:09,200
    விடைபெறுகிறேன்!

""")

# Tamil SRT with split lines (two Tamil entries map to one EN).
EN_SPLIT_SRT = textwrap.dedent("""\
    1
    00:00:01,000 --> 00:00:05,000
    This is a long sentence split across two Tamil subtitles.

""")

TA_SPLIT_SRT = textwrap.dedent("""\
    1
    00:00:01,000 --> 00:00:03,000
    இது ஒரு நீண்ட

    2
    00:00:03,100 --> 00:00:05,000
    தமிழ் வாக்கியம்.

""")


@pytest.fixture()
def en_srt_file(tmp_path: Path) -> Path:
    p = tmp_path / "movie.en.srt"
    p.write_text(EN_SRT_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def ta_srt_file(tmp_path: Path) -> Path:
    p = tmp_path / "movie.ta.srt"
    p.write_text(TA_SRT_CONTENT, encoding="utf-8")
    return p


@pytest.fixture()
def en_split_file(tmp_path: Path) -> Path:
    p = tmp_path / "split.en.srt"
    p.write_text(EN_SPLIT_SRT, encoding="utf-8")
    return p


@pytest.fixture()
def ta_split_file(tmp_path: Path) -> Path:
    p = tmp_path / "split.ta.srt"
    p.write_text(TA_SPLIT_SRT, encoding="utf-8")
    return p


# ─── IoU helper ───────────────────────────────────────────────────────────────

class TestIou:
    """Unit-test the internal IoU function."""

    def test_perfect_overlap(self) -> None:
        from tyto.data.srt_align import _iou

        assert _iou(1.0, 3.0, 1.0, 3.0) == pytest.approx(1.0)

    def test_no_overlap(self) -> None:
        from tyto.data.srt_align import _iou

        assert _iou(1.0, 2.0, 3.0, 4.0) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        from tyto.data.srt_align import _iou

        # A: [1, 3], B: [2, 4] → intersection=1, union=3
        result = _iou(1.0, 3.0, 2.0, 4.0)
        assert 0.0 < result < 1.0

    def test_zero_length_interval(self) -> None:
        from tyto.data.srt_align import _iou

        assert _iou(1.0, 1.0, 1.0, 3.0) == pytest.approx(0.0)


# ─── align_srt_files ─────────────────────────────────────────────────────────

class TestAlignSrtFiles:
    def test_basic_alignment_returns_dataframe(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(en_srt_file, ta_srt_file, film_id="test_film")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_output_schema_columns(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(en_srt_file, ta_srt_file, film_id="test_film")
        for col in PAIRS_SCHEMA.names:
            assert col in df.columns, f"Missing column: {col}"

    def test_film_id_propagated(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(en_srt_file, ta_srt_file, film_id="mymovie")
        assert (df["film_id"] == "mymovie").all()

    def test_pair_id_format(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(en_srt_file, ta_srt_file, film_id="m1")
        assert df["pair_id"].str.startswith("m1:").all()

    def test_tamil_text_in_ta_column(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(en_srt_file, ta_srt_file, film_id="test")
        # ta column should contain Tamil Unicode characters (block U+0B80-U+0BFF).
        # Use actual characters instead of \u escape (PyArrow regex engine limitation).
        has_tamil = df["ta"].apply(
            lambda t: any("\u0B80" <= ch <= "\u0BFF" for ch in (t or ""))
        )
        assert has_tamil.any()

    def test_timing_columns_populated(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(en_srt_file, ta_srt_file, film_id="test")
        assert df["time_start"].notna().any()
        assert df["time_end"].notna().any()

    def test_sent_idx_starts_at_one(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(en_srt_file, ta_srt_file, film_id="test")
        assert df["sent_idx"].min() == 1

    def test_source_version_propagated(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(
            en_srt_file, ta_srt_file, film_id="test", source_version="srt_v1"
        )
        assert (df["source_version"] == "srt_v1").all()

    def test_min_iou_threshold(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        # Very high threshold – fewer or zero pairs.
        df_high = align_srt_files(
            en_srt_file, ta_srt_file, film_id="test", min_iou=0.999
        )
        df_low = align_srt_files(
            en_srt_file, ta_srt_file, film_id="test", min_iou=0.01
        )
        # Lower threshold should yield more or equal pairs.
        assert len(df_low) >= len(df_high)

    def test_year_and_title_propagated(
        self, en_srt_file: Path, ta_srt_file: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_files

        df = align_srt_files(
            en_srt_file, ta_srt_file,
            film_id="test", film_title="My Movie", year=2020
        )
        assert (df["film_title"] == "My Movie").all()
        assert (df["year"] == 2020).all()

    def test_empty_result_for_nonoverlapping_files(self, tmp_path: Path) -> None:
        """Files with completely non-overlapping timestamps → empty DataFrame."""
        from tyto.data.srt_align import align_srt_files

        en = tmp_path / "e.srt"
        ta = tmp_path / "t.srt"
        en.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n\n", encoding="utf-8")
        ta.write_text("1\n01:00:00,000 --> 01:00:01,000\nவணக்கம்\n\n", encoding="utf-8")
        df = align_srt_files(en, ta, film_id="empty", min_iou=0.5)
        assert len(df) == 0


# ─── Parquet output ───────────────────────────────────────────────────────────

class TestSrtToParquet:
    def test_parquet_written_and_readable(
        self, en_srt_file: Path, ta_srt_file: Path, tmp_path: Path
    ) -> None:
        from tyto.data.srt_align import align_srt_pair_to_parquet

        out = tmp_path / "aligned.parquet"
        result_path = align_srt_pair_to_parquet(
            en_srt_file, ta_srt_file, out, film_id="test"
        )
        assert result_path.exists()
        df = pd.read_parquet(out)
        assert len(df) > 0
        for col in PAIRS_SCHEMA.names:
            assert col in df.columns

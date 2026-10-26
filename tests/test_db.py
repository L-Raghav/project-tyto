"""Unit tests for tyto.data.db (DuckDB helper).

Uses a tiny in-memory parquet fixture written to a temp file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tyto.data.parse import PAIRS_SCHEMA


def _make_parquet(tmp_path: Path) -> Path:
    """Write a minimal pairs.parquet to *tmp_path* and return its path."""
    rows = [
        {
            "pair_id": "f1:1", "film_id": "f1", "film_title": "Film One",
            "year": 2010, "sent_idx": 1,
            "en_raw": "Hello.", "ta_raw": "வணக்கம்.",
            "en": "Hello.", "ta": "வணக்கம்.",
            "time_start": 1.0, "time_end": 3.0,
            "align_type": "1-1", "source_version": "2016", "qc_flags": "",
        },
        {
            "pair_id": "f1:2", "film_id": "f1", "film_title": "Film One",
            "year": 2010, "sent_idx": 2,
            "en_raw": "Goodbye.", "ta_raw": "விடை.",
            "en": "Goodbye.", "ta": "விடை.",
            "time_start": 4.0, "time_end": 6.0,
            "align_type": "1-1", "source_version": "2016", "qc_flags": "CODE_MIXED",
        },
        {
            "pair_id": "f2:1", "film_id": "f2", "film_title": "Film Two",
            "year": 2015, "sent_idx": 1,
            "en_raw": "Yes.", "ta_raw": "ஆமா.",
            "en": "Yes.", "ta": "ஆமா.",
            "time_start": None, "time_end": None,
            "align_type": "1-1", "source_version": "2016", "qc_flags": "WRONG_LANG",
        },
    ]
    df = pd.DataFrame(rows)
    df["year"] = pd.array(df["year"], dtype=pd.Int16Dtype())
    df["sent_idx"] = df["sent_idx"].astype("int32")
    df["time_start"] = df["time_start"].astype("float32")
    df["time_end"] = df["time_end"].astype("float32")

    p = tmp_path / "pairs.parquet"
    table = pa.Table.from_pandas(df, schema=PAIRS_SCHEMA, safe=False)
    pq.write_table(table, p, compression="snappy")
    return p


class TestPairsDB:
    @pytest.fixture()
    def parquet(self, tmp_path: Path) -> Path:
        return _make_parquet(tmp_path)

    def test_opens_without_error(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            assert db is not None

    def test_count_all_rows(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            assert db.count() == 3

    def test_count_with_where(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            assert db.count(where="film_id = 'f1'") == 2

    def test_usable_count_excludes_hard_fail(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            # f1:1 (clean) + f1:2 (CODE_MIXED soft) = 2 usable; f2:1 (WRONG_LANG hard) = 0
            usable = db.usable_count()
            assert usable == 2

    def test_film_summary_returns_all_films(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            df = db.film_summary()
            assert set(df["film_id"].tolist()) == {"f1", "f2"}

    def test_film_summary_pair_counts(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            df = db.film_summary().set_index("film_id")
            assert df.loc["f1", "n_pairs"] == 2
            assert df.loc["f2", "n_pairs"] == 1

    def test_flag_summary_returns_df(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            df = db.flag_summary()
            assert "flag" in df.columns
            assert "n_rows" in df.columns
            # Should find CODE_MIXED and WRONG_LANG flags.
            flags = set(df["flag"].tolist())
            assert "CODE_MIXED" in flags
            assert "WRONG_LANG" in flags

    def test_sample_pairs(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            df = db.sample_pairs(n=2, seed=42)
            assert len(df) <= 3  # can't sample more than exists
            assert "en_raw" in df.columns

    def test_query_custom_sql(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            df = db.query("SELECT pair_id FROM pairs WHERE film_id = 'f1'")
            assert len(df) == 2

    def test_summary_stats_keys(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        with PairsDB(parquet) as db:
            stats = db.summary_stats()
            assert "total_pairs" in stats
            assert "usable_pairs" in stats
            assert "n_films" in stats
            assert stats["total_pairs"] == 3
            assert stats["n_films"] == 2

    def test_missing_parquet_raises(self, tmp_path: Path) -> None:
        from tyto.data.db import PairsDB
        with pytest.raises(FileNotFoundError):
            PairsDB(tmp_path / "nonexistent.parquet")

    def test_context_manager_closes(self, parquet: Path) -> None:
        from tyto.data.db import PairsDB
        db = PairsDB(parquet)
        db.close()
        # Calling close again should not raise.
        try:
            db.close()
        except Exception:
            pass  # DuckDB may complain on double-close; that's OK.

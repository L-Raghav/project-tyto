"""DuckDB helper for querying pairs.parquet.

Provides a lightweight wrapper around DuckDB that:
- Opens a connection once (in-memory or file-backed).
- Registers pairs.parquet as a virtual table named ``pairs``.
- Offers common query helpers and a summary report.

Usage::

    from tyto.data.db import PairsDB
    db = PairsDB("data/processed/pairs.parquet")
    df = db.query(
        "SELECT film_id, COUNT(*) AS n FROM pairs "
        "GROUP BY film_id ORDER BY n DESC LIMIT 10"
    )
    db.close()

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class PairsDB:
    """DuckDB helper for querying ``pairs.parquet``.

    Args:
        parquet_path: Path to the pairs.parquet file.
        db_path:      Optional path for a persistent DuckDB file.  If omitted,
                      an in-memory database is used (faster, no disk overhead).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        db_path: str | None = None,
    ) -> None:
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.parquet_path}")

        self._conn = duckdb.connect(database=db_path or ":memory:")
        # Register the parquet file as a view so no data is copied into memory.
        self._conn.execute(
            "CREATE OR REPLACE VIEW pairs AS "
            f"SELECT * FROM read_parquet('{self.parquet_path.as_posix()}')"
        )
        logger.info("PairsDB initialised from %s", self.parquet_path)

    # ─── Core query ─────────────────────────────────────────────────────────

    def query(self, sql: str, **params: Any) -> pd.DataFrame:
        """Run a SQL query against the ``pairs`` view and return a DataFrame.

        Args:
            sql:    SQL query string.  Use ``pairs`` as the table name.
            params: Named parameters for parameterised queries (passed to
                    DuckDB's ``execute``).

        Returns:
            Pandas DataFrame of the result.

        Example::

            db.query(
                "SELECT * FROM pairs WHERE film_id = $film_id LIMIT 5",
                film_id="12345"
            )
        """
        if params:
            rel = self._conn.execute(sql, list(params.values()))
        else:
            rel = self._conn.execute(sql)
        return rel.df()

    # ─── Convenience helpers ─────────────────────────────────────────────────

    def count(self, where: str = "") -> int:
        """Return the total number of rows (optionally filtered).

        Args:
            where: Optional SQL WHERE clause (without the ``WHERE`` keyword).

        Returns:
            Row count as int.
        """
        clause = f"WHERE {where}" if where else ""
        result = self._conn.execute(f"SELECT COUNT(*) FROM pairs {clause}").fetchone()
        return int(result[0]) if result else 0

    def usable_count(self) -> int:
        """Return the number of rows with no hard-fail QC flags."""
        hard_flags = ["TOO_SHORT_EN", "TOO_SHORT_TA", "TOO_LONG_EN", "TOO_LONG_TA",
                      "BAD_RATIO", "WRONG_LANG"]
        conditions = " AND ".join(
            [f"qc_flags NOT LIKE '%{f}%'" for f in hard_flags]
        )
        return self.count(where=conditions)

    def film_summary(self) -> pd.DataFrame:
        """Return a per-film summary: pair count, unique sentences, year.

        Returns:
            DataFrame with columns ``film_id``, ``film_title``, ``year``,
            ``n_pairs``.
        """
        return self.query(
            """
            SELECT
                film_id,
                MAX(film_title) AS film_title,
                MAX(year) AS year,
                COUNT(*) AS n_pairs
            FROM pairs
            GROUP BY film_id
            ORDER BY n_pairs DESC
            """
        )

    def flag_summary(self) -> pd.DataFrame:
        """Return per-flag counts by unnesting the pipe-separated qc_flags column.

        Returns:
            DataFrame with columns ``flag``, ``n_rows``.
        """
        return self.query(
            """
            SELECT
                flag,
                COUNT(*) AS n_rows
            FROM (
                SELECT UNNEST(string_split(qc_flags, '|')) AS flag
                FROM pairs
                WHERE qc_flags != ''
            )
            WHERE flag != ''
            GROUP BY flag
            ORDER BY n_rows DESC
            """
        )

    def sample_pairs(self, n: int = 30, seed: int = 42) -> pd.DataFrame:
        """Return *n* random pairs for human review.

        Args:
            n:    Number of pairs to return.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with key columns for review.
        """
        return self.query(
            f"""
            SELECT
                pair_id, film_title, year, sent_idx,
                en_raw, ta_raw, en, ta,
                align_type, qc_flags
            FROM pairs
            USING SAMPLE {n} ROWS (reservoir, {seed})
            """
        )

    def length_distribution(self) -> pd.DataFrame:
        """Return per-row character lengths for distribution plotting."""
        return self.query(
            """
            SELECT
                LENGTH(en) AS en_len,
                LENGTH(ta) AS ta_len
            FROM pairs
            """
        )

    def summary_stats(self) -> dict[str, Any]:
        """Return a dict of summary statistics for the audit report."""
        total = self.count()
        usable = self.usable_count()
        film_n = self.query("SELECT COUNT(DISTINCT film_id) AS n FROM pairs").iloc[0]["n"]
        return {
            "total_pairs": total,
            "usable_pairs": usable,
            "n_films": int(film_n),
            "usable_pct": round(100.0 * usable / max(total, 1), 2),
        }

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
        logger.debug("PairsDB connection closed")

    def __enter__(self) -> PairsDB:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

"""Unit tests for tyto.data.qc.

Tests cover QC pipeline steps on small in-memory fixtures.
All tests run on CPU with no network access in under 60 seconds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tyto.utils.config import TytoConfig


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal pairs DataFrame from a list of row dicts."""
    defaults = {
        "pair_id": "x:1",
        "film_id": "x",
        "film_title": "",
        "year": None,
        "sent_idx": 1,
        "en_raw": "",
        "ta_raw": "",
        "en": "",
        "ta": "",
        "time_start": None,
        "time_end": None,
        "align_type": "1-1",
        "source_version": "test",
        "qc_flags": "",
    }
    records = [{**defaults, **r} for r in rows]
    df = pd.DataFrame(records)
    df["year"] = pd.array(df["year"], dtype=pd.Int16Dtype())
    df["sent_idx"] = df["sent_idx"].astype("int32")
    df["time_start"] = df["time_start"].astype("float32")
    df["time_end"] = df["time_end"].astype("float32")
    return df


@pytest.fixture()
def cfg() -> TytoConfig:
    """Return a TytoConfig with a very low min_usable_pairs to avoid stop-gate."""
    c = TytoConfig()
    c.qc.min_usable_pairs = 1
    return c


# ─── Length filter ────────────────────────────────────────────────────────────

class TestLengthFilters:
    def test_too_short_en_flagged(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_length_filters

        cfg.qc.min_en_tokens = 2
        df = _make_df([{"en": "Hi", "ta": "வணக்கம் இது சோதனை"}])
        result = apply_length_filters(df, cfg)
        assert "TOO_SHORT_EN" in result.iloc[0]["qc_flags"]

    def test_too_long_en_flagged(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_length_filters

        cfg.qc.max_en_tokens = 3
        long_text = " ".join(["word"] * 10)
        df = _make_df([{"en": long_text, "ta": "வணக்கம்"}])
        result = apply_length_filters(df, cfg)
        assert "TOO_LONG_EN" in result.iloc[0]["qc_flags"]

    def test_bad_length_ratio_flagged(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_length_filters

        cfg.qc.max_length_ratio = 3.0
        # en = 5 chars, ta = 100 chars → ratio ≈ 20
        df = _make_df([{"en": "Hello", "ta": "அ" * 100}])
        result = apply_length_filters(df, cfg)
        assert "BAD_RATIO" in result.iloc[0]["qc_flags"]

    def test_clean_pair_no_flags(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_length_filters

        df = _make_df([{"en": "Hello world", "ta": "வணக்கம் உலகம்"}])
        result = apply_length_filters(df, cfg)
        assert result.iloc[0]["qc_flags"] == ""


# ─── Script flags ─────────────────────────────────────────────────────────────

class TestScriptFlags:
    def test_pure_english_ta_flagged_wrong_lang(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_script_flags

        df = _make_df([{"en": "Hello", "ta": "Hello world again and again."}])
        result = apply_script_flags(df, cfg)
        assert "WRONG_LANG" in result.iloc[0]["qc_flags"]

    def test_pure_tamil_no_flag(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_script_flags

        df = _make_df([{"en": "Hello", "ta": "வணக்கம் உலகம் அனைவருக்கும்"}])
        result = apply_script_flags(df, cfg)
        assert result.iloc[0]["qc_flags"] == ""

    def test_romanised_ta_flagged(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_script_flags

        cfg.qc.tamil_ratio_min = 0.4
        cfg.qc.ascii_ratio_romanised = 0.7
        df = _make_df([{"en": "Hello", "ta": "vanakkam makkal ellorum"}])
        result = apply_script_flags(df, cfg)
        flags = result.iloc[0]["qc_flags"]
        assert "ROMANISED_TA" in flags or "WRONG_LANG" in flags


# ─── Exact dedup ─────────────────────────────────────────────────────────────

class TestExactDedup:
    def test_unique_pairs_count_one(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_exact_dedup

        df = _make_df([
            {"pair_id": "a:1", "en": "Hello", "ta": "வணக்கம்"},
            {"pair_id": "b:2", "en": "World", "ta": "உலகம்"},
        ])
        result = apply_exact_dedup(df)
        assert (result["repeat_count"] == 1).all()
        assert (result["qc_flags"] == "").all()

    def test_duplicate_pairs_flagged(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_exact_dedup

        df = _make_df([
            {"pair_id": "a:1", "en": "Yes", "ta": "ஆமா"},
            {"pair_id": "b:2", "en": "Yes", "ta": "ஆமா"},
            {"pair_id": "c:3", "en": "No",  "ta": "இல்லை"},
        ])
        result = apply_exact_dedup(df)
        dup_rows = result[result["en"] == "Yes"]
        assert (dup_rows["repeat_count"] == 2).all()
        assert all("EXACT_DUP" in f for f in dup_rows["qc_flags"])
        # The unique row should not be flagged.
        unique_row = result[result["en"] == "No"]
        assert unique_row.iloc[0]["repeat_count"] == 1

    def test_repeat_count_column_type(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_exact_dedup

        df = _make_df([{"en": "Hi", "ta": "வணக்கம்"}])
        result = apply_exact_dedup(df)
        assert "repeat_count" in result.columns


# ─── Near dedup ───────────────────────────────────────────────────────────────

class TestNearDedup:
    def test_identical_en_flagged_near_dup(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_near_dedup

        cfg.qc.minhash_num_perm = 32  # Fast for tests.
        cfg.qc.minhash_threshold = 0.85
        # Two nearly identical sentences.
        df = _make_df([
            {"pair_id": "a:1", "en": "The quick brown fox jumps over the lazy dog.",
             "ta": "வணக்கம்"},
            {"pair_id": "a:2", "en": "The quick brown fox jumps over the lazy dog.",
             "ta": "நன்றி"},
        ])
        result = apply_near_dedup(df, cfg)
        assert "repeat_count" not in result.columns  # near_dedup doesn't add this
        flags = result["qc_flags"].tolist()
        assert any("NEAR_DUP" in f for f in flags)

    def test_distinct_sentences_not_flagged(self, cfg: TytoConfig) -> None:
        from tyto.data.qc import apply_near_dedup

        cfg.qc.minhash_num_perm = 32
        cfg.qc.minhash_threshold = 0.85
        df = _make_df([
            {"pair_id": "a:1", "en": "Hello world", "ta": "வணக்கம்"},
            {"pair_id": "a:2", "en": "Goodbye forever", "ta": "நன்றி"},
        ])
        result = apply_near_dedup(df, cfg)
        # With clearly distinct texts, no flags expected.
        assert all("NEAR_DUP" not in f for f in result["qc_flags"])


# ─── Stop-gate ────────────────────────────────────────────────────────────────

class TestStopGate:
    def test_stop_gate_raises_when_below_threshold(self) -> None:
        from tyto.data.qc import run_qc_pipeline

        cfg = TytoConfig()
        cfg.qc.min_usable_pairs = 100_000  # impossibly high
        df = _make_df([{"en": "Hello", "ta": "வணக்கம்"}])
        with pytest.raises(RuntimeError, match="STOP GATE"):
            run_qc_pipeline(df, cfg)

    def test_pipeline_passes_when_above_threshold(self) -> None:
        from tyto.data.qc import run_qc_pipeline

        cfg = TytoConfig()
        cfg.qc.min_usable_pairs = 1
        df = _make_df([{"en": "Hello world", "ta": "வணக்கம் உலகம்"}])
        result = run_qc_pipeline(df, cfg)
        assert len(result) == 1


# ─── usable_count helper ─────────────────────────────────────────────────────

class TestUsableCount:
    def test_hard_fail_not_counted(self) -> None:
        from tyto.data.qc import count_usable

        df = _make_df([
            {"en": "Hello", "ta": "வணக்கம்", "qc_flags": ""},
            {"en": "Hi",    "ta": "நன்றி",   "qc_flags": "WRONG_LANG"},
            {"en": "Hey",   "ta": "ஆமா",     "qc_flags": "TOO_SHORT_EN"},
        ])
        assert count_usable(df) == 1  # Only the clean row.

    def test_soft_flags_still_usable(self) -> None:
        from tyto.data.qc import count_usable

        df = _make_df([
            {"en": "Hello", "ta": "வணக்கம்", "qc_flags": "NEAR_DUP"},
            {"en": "Hi",    "ta": "நன்றி",   "qc_flags": "CODE_MIXED"},
        ])
        assert count_usable(df) == 2

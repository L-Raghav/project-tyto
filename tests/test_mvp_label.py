"""Unit tests for Project Tyto MVP Stage 4 heuristic risk scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from tyto.mvp.label import SIGNAL_WEIGHTS, compute_risk_scores


def _make_synthetic_features(n: int = 100) -> pd.DataFrame:
    """Create a synthetic feature DataFrame with smooth continuous variation."""
    rng = np.random.default_rng(42)
    # Distinct labse_cos values ensure unique risk scores and crisp percentile cuts
    labse_cos = np.linspace(0.10, 0.95, n)
    len_ratio = np.linspace(1.0, 12.0, n)
    return pd.DataFrame(
        {
            "pair_id": np.arange(n),
            "en": [f"Line {i} with 5 items" if i % 5 == 0 else f"Line {i}" for i in range(n)],
            "ta": [f"வரி {i}" for i in range(n)],
            "labse_cos": labse_cos,
            "len_ratio": len_ratio,
            "ta_spaced_punct": np.where(np.arange(n) % 10 == 0, 0.05, 0.0),
            "ta_latin_share": np.where(np.arange(n) % 8 == 0, 0.25, 0.01),
            "digit_match": rng.integers(0, 2, size=n),
            "punct_parity": np.where(np.arange(n) % 7 == 0, 1, 0),
            "idiom_present": np.where(np.arange(n) % 12 == 0, 1, 0),
        }
    )


def test_risk_tiers() -> None:
    """Verify High = top 10%, Medium = next 20%, Low = remaining 70%."""
    df = _make_synthetic_features(100)
    # Remove binary noise so all 100 risk_scores are strictly distinct
    df["ta_spaced_punct"] = 0.0
    df["ta_latin_share"] = 0.0
    df["digit_match"] = 1
    df["punct_parity"] = 0
    df["idiom_present"] = 0
    df["len_ratio"] = 5.0  # identical -> no extreme flags

    scored = compute_risk_scores(df)
    counts = scored["risk_tier"].value_counts().to_dict()
    assert counts["High"] == 10
    assert counts["Medium"] == 20
    assert counts["Low"] == 70

    high_min = scored.loc[scored["risk_tier"] == "High", "risk_score"].min()
    med_max = scored.loc[scored["risk_tier"] == "Medium", "risk_score"].max()
    med_min = scored.loc[scored["risk_tier"] == "Medium", "risk_score"].min()
    low_max = scored.loc[scored["risk_tier"] == "Low", "risk_score"].max()
    assert high_min >= med_max
    assert med_min >= low_max


def test_risk_score_bounded() -> None:
    """risk_score must always stay within [0, 100], including NaN labse_cos."""
    df = _make_synthetic_features(50)
    scored = compute_risk_scores(df)
    assert (scored["risk_score"] >= 0.0).all()
    assert (scored["risk_score"] <= 100.0).all()

    df_nan = df.copy()
    df_nan["labse_cos"] = np.nan
    scored_nan = compute_risk_scores(df_nan)
    assert (scored_nan["risk_score"] >= 0.0).all()
    assert (scored_nan["risk_score"] <= 100.0).all()


def test_risk_all_signals() -> None:
    """All 7 risk signals compute without error and lie in [0, 1]."""
    df = _make_synthetic_features(40)
    scored = compute_risk_scores(df)
    for col in SIGNAL_WEIGHTS:
        assert col in scored.columns
        assert scored[col].notna().all()
        assert (scored[col] >= 0.0).all()
        assert (scored[col] <= 1.0).all()

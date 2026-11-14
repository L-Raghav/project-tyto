"""Heuristic risk scoring for Project Tyto MVP (Stage 4).

Computes a 0-100 heuristic risk proxy from weak surface and similarity signals
and assigns High / Medium / Low risk tiers by corpus percentile.

NOTE: Weights are arbitrary and not derived from human quality labels.
Risk scores are heuristic proxies only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Arbitrary heuristic weights (sum to 100). Not derived from human labels.
SIGNAL_WEIGHTS: dict[str, float] = {
    "sig_labse_cos_inv_rank": 30.0,
    "sig_len_ratio_extreme": 15.0,
    "sig_ta_spaced_punct": 10.0,
    "sig_ta_latin_heavy": 10.0,
    "sig_digit_mismatch": 10.0,
    "sig_punct_parity": 10.0,
    "sig_idiom_low_sim": 15.0,
}


def compute_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-signal heuristic values, ``risk_score`` (0-100), and ``risk_tier``.

    Parameters
    ----------
    df:
        Input DataFrame containing feature columns from ``extract_features``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with per-signal columns, ``risk_score``, and ``risk_tier``.
    """
    out = df.copy()

    # 1. Inverted rank-normalized LaBSE cosine similarity (weight 30)
    cos = out["labse_cos"].astype(float)
    if cos.notna().any():
        pct_rank = cos.rank(pct=True, method="average")
        inv_rank = (1.0 - pct_rank).fillna(0.5)
    else:
        inv_rank = pd.Series(0.5, index=out.index, dtype=float)
    out["sig_labse_cos_inv_rank"] = inv_rank.astype(float)

    # 2. Extreme length ratio (below p5 or above p95) (weight 15)
    len_ratio = out["len_ratio"].astype(float)
    p5 = float(np.nanpercentile(len_ratio, 5))
    p95 = float(np.nanpercentile(len_ratio, 95))
    out["sig_len_ratio_extreme"] = ((len_ratio < p5) | (len_ratio > p95)).astype(float)

    # 3. Spaced punctuation in Tamil (weight 10)
    out["sig_ta_spaced_punct"] = (out["ta_spaced_punct"].astype(float) > 0.0).astype(float)

    # 4. Latin-heavy Tamil text (weight 10)
    out["sig_ta_latin_heavy"] = (out["ta_latin_share"].astype(float) > 0.1).astype(float)

    # 5. Digit mismatch when digits are present in English (weight 10)
    has_en_digit = out["en"].fillna("").astype(str).str.contains(r"\d", regex=True)
    out["sig_digit_mismatch"] = (
        has_en_digit & (out["digit_match"].astype(int) == 0)
    ).astype(float)

    # 6. Punctuation count mismatch (weight 10)
    out["sig_punct_parity"] = (out["punct_parity"].astype(int) > 0).astype(float)

    # 7. Idiom present AND LaBSE cosine below corpus median (weight 15)
    if cos.notna().any():
        cos_median = float(cos.median())
        below_median = cos.notna() & (cos < cos_median)
    else:
        below_median = pd.Series(False, index=out.index)
    out["sig_idiom_low_sim"] = (
        (out["idiom_present"].astype(int) == 1) & below_median
    ).astype(float)

    # Weighted sum (0-100)
    score = pd.Series(0.0, index=out.index, dtype=float)
    for col, weight in SIGNAL_WEIGHTS.items():
        score = score + weight * out[col]
    out["risk_score"] = score

    # Risk tiers by percentile of risk_score: High >= p90, Medium >= p70, Low < p70
    p70 = float(np.percentile(score, 70))
    p90 = float(np.percentile(score, 90))
    tiers = np.where(
        score >= p90,
        "High",
        np.where(score >= p70, "Medium", "Low"),
    )
    out["risk_tier"] = tiers
    return out


def score_risk(
    features_path: str | Path,
    out_path: str | Path,
) -> pd.DataFrame:
    """Load features parquet, compute heuristic risk scores, and write risk.parquet."""
    features_path = Path(features_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)
    risk_df = compute_risk_scores(df)

    logger.info("Writing risk parquet to %s", out_path)
    pq.write_table(pa.Table.from_pandas(risk_df), str(out_path))

    tier_counts = risk_df["risk_tier"].value_counts().to_dict()
    print(
        f"score_risk complete: {len(risk_df):,} rows, "
        f"mean={risk_df['risk_score'].mean():.2f}, "
        f"p70={risk_df['risk_score'].quantile(0.70):.2f}, "
        f"p90={risk_df['risk_score'].quantile(0.90):.2f}, "
        f"tiers={tier_counts} → {out_path}"
    )
    return risk_df

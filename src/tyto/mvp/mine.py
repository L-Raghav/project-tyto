"""Pattern mining for Project Tyto MVP (Stage 5).

Runs:
(a) FP-Growth association rules (mlxtend) -> artifacts/association_rules.parquet
(b) KMeans clustering (k=3..8 by silhouette) -> artifacts/clusters.parquet
(c) IsolationForest top-30 anomalies -> artifacts/anomalies.parquet
(d) Per-film summary table -> artifacts/film_stats.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CLUSTER_FEATURES: list[str] = [
    "en_words",
    "ta_chars",
    "len_ratio",
    "ta_vowel_share",
    "ta_latin_share",
    "ta_spaced_punct",
    "en_exclaim",
    "en_question",
    "en_caps_ratio",
    "digit_match",
    "punct_parity",
    "repeat_count",
    "is_short",
    "ta_hedge_count",
    "idiom_present",
    "risk_score",
    "labse_cos",
]


def _build_fpgrowth_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build boolean transaction matrix for FP-Growth."""
    items = pd.DataFrame(index=df.index)

    # Bin en_words into [1-3, 4-7, 8-15, 16+]
    en_word_bins = pd.cut(
        df["en_words"].astype(float),
        bins=[-1, 3, 7, 15, float("inf")],
        labels=["1-3", "4-7", "8-15", "16+"],
    )
    for lbl in ["1-3", "4-7", "8-15", "16+"]:
        items[f"en_words={lbl}"] = (en_word_bins == lbl).fillna(False).to_numpy(dtype=bool)

    # Bin len_ratio into [low, medium, high, very_high] by quartiles
    try:
        lr_bins = pd.qcut(
            df["len_ratio"].astype(float),
            q=4,
            labels=["low", "medium", "high", "very_high"],
            duplicates="drop",
        )
        for lbl in lr_bins.cat.categories:
            items[f"len_ratio={lbl}"] = (lr_bins == lbl).fillna(False).to_numpy(dtype=bool)
    except ValueError:
        items["len_ratio=all"] = True

    # Bin ta_spaced_punct into [none, some]
    has_spaced = df["ta_spaced_punct"].astype(float) > 0.0
    items["ta_spaced_punct=none"] = (~has_spaced).to_numpy(dtype=bool)
    items["ta_spaced_punct=some"] = has_spaced.to_numpy(dtype=bool)

    # Bin ta_latin_share into [low, high] by median
    latin = df["ta_latin_share"].astype(float)
    latin_med = float(latin.median())
    is_latin_high = latin > latin_med
    items["ta_latin_share=low"] = (~is_latin_high).to_numpy(dtype=bool)
    items["ta_latin_share=high"] = is_latin_high.to_numpy(dtype=bool)

    # Binary indicators
    items["is_short"] = (df["is_short"].astype(int) == 1).to_numpy(dtype=bool)
    items["en_exclaim"] = (df["en_exclaim"].astype(int) == 1).to_numpy(dtype=bool)
    items["en_question"] = (df["en_question"].astype(int) == 1).to_numpy(dtype=bool)
    items["en_caps_high"] = (df["en_caps_ratio"].astype(float) > 0.5).to_numpy(dtype=bool)
    items["idiom_present"] = (df["idiom_present"].astype(int) == 1).to_numpy(dtype=bool)

    if "sig_digit_mismatch" in df.columns:
        items["digit_mismatch"] = (df["sig_digit_mismatch"].astype(float) > 0).to_numpy(dtype=bool)
    else:
        has_digit = df["en"].fillna("").astype(str).str.contains(r"\d", regex=True)
        items["digit_mismatch"] = (has_digit & (df["digit_match"].astype(int) == 0)).to_numpy(
            dtype=bool
        )

    # Risk tier one-hot
    for tier in ("High", "Medium", "Low"):
        items[f"risk_tier={tier}"] = (df["risk_tier"] == tier).to_numpy(dtype=bool)

    # Film ID one-hot
    for fid in sorted(df["film_id"].astype(str).unique()):
        items[f"film_id={fid}"] = (df["film_id"].astype(str) == fid).to_numpy(dtype=bool)

    return items


def _run_fpgrowth(df: pd.DataFrame) -> pd.DataFrame:
    """Mine association rules with min_support 0.01 -> 0.005 fallback."""
    tx = _build_fpgrowth_matrix(df)
    rules = pd.DataFrame()
    used_support = 0.01

    for min_sup in (0.01, 0.005):
        used_support = min_sup
        freq = fpgrowth(tx, min_support=min_sup, use_colnames=True, max_len=4)
        if len(freq) == 0:
            continue
        candidate_rules = association_rules(freq, metric="lift", min_threshold=1.0)
        if len(candidate_rules) > 0:
            rules = candidate_rules
            break

    if len(rules) == 0:
        logger.warning("No association rules found at min_support=0.005")
        return pd.DataFrame(
            columns=[
                "antecedents",
                "consequents",
                "antecedent support",
                "consequent support",
                "support",
                "confidence",
                "lift",
                "min_support_used",
            ]
        )

    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(15).copy()
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules["min_support_used"] = used_support
    cols = [
        "antecedents",
        "consequents",
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "min_support_used",
    ]
    return rules[cols].reset_index(drop=True)


def _prepare_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Standardize CLUSTER_FEATURES, filling NaN labse_cos with corpus median."""
    mat = df[CLUSTER_FEATURES].copy()
    if mat["labse_cos"].isna().all():
        mat["labse_cos"] = 0.0
    else:
        mat["labse_cos"] = mat["labse_cos"].fillna(float(mat["labse_cos"].median()))
    mat = mat.fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(mat.to_numpy(dtype=np.float64))


def _run_kmeans(
    df: pd.DataFrame, x_scaled: np.ndarray
) -> tuple[pd.DataFrame, int, float]:
    """Fit KMeans for k=3..8, pick best by silhouette score, mark 5 exemplars per cluster."""
    n = len(df)
    sample_sz = min(10_000, n)

    best_k = 3
    best_sil = -1.0
    best_model: KMeans | None = None
    best_labels: np.ndarray | None = None

    for k in range(3, 9):
        if k >= n:
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(x_scaled)
        if len(np.unique(labels)) < 2:
            continue
        sil = float(
            silhouette_score(
                x_scaled,
                labels,
                sample_size=sample_sz if n > sample_sz else None,
                random_state=42,
            )
        )
        logger.info("KMeans k=%d silhouette=%.4f", k, sil)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_model = km
            best_labels = labels

    if best_model is None or best_labels is None:
        best_model = KMeans(n_clusters=min(3, n), random_state=42, n_init=10)
        best_labels = best_model.fit_predict(x_scaled)
        best_k = len(np.unique(best_labels))
        best_sil = 0.0

    # Distance to assigned centroid in standardized space
    centers = best_model.cluster_centers_
    dists = np.linalg.norm(x_scaled - centers[best_labels], axis=1)

    out = df.copy()
    out["cluster_id"] = best_labels.astype(int)
    out["dist_to_centroid"] = dists.astype(float)
    out["is_exemplar"] = False

    # Mark 5 closest pairs per cluster as exemplars
    for cid in sorted(out["cluster_id"].unique()):
        idx = out.index[out["cluster_id"] == cid]
        top5 = out.loc[idx, "dist_to_centroid"].nsmallest(5).index
        out.loc[top5, "is_exemplar"] = True

    return out, best_k, best_sil


def _run_isolation_forest(df: pd.DataFrame, x_scaled: np.ndarray) -> pd.DataFrame:
    """Fit IsolationForest and return top-30 most anomalous pairs."""
    iso = IsolationForest(random_state=42, contamination="auto", n_jobs=-1)
    iso.fit(x_scaled)
    scores = iso.decision_function(x_scaled)

    out = df.copy()
    out["anomaly_score"] = scores.astype(float)
    top30 = out.nsmallest(min(30, len(out)), "anomaly_score").reset_index(drop=True)
    return top30


def _build_film_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-film statistics."""
    grouped = (
        df.assign(_has_spaced=(df["ta_spaced_punct"].astype(float) > 0.0).astype(float))
        .groupby("film_id", as_index=False)
        .agg(
            pairs=("pair_id", "count"),
            mean_risk=("risk_score", "mean"),
            spaced_punct_rate=("_has_spaced", "mean"),
            latin_share=("ta_latin_share", "mean"),
            mean_len_ratio=("len_ratio", "mean"),
        )
        .sort_values("mean_risk", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def run_mining(
    risk_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Execute FP-Growth, KMeans, IsolationForest, and per-film aggregation."""
    risk_path = Path(risk_path)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading risk data from %s", risk_path)
    df = pd.read_parquet(risk_path)

    logger.info("Mining FP-Growth association rules ...")
    rules_df = _run_fpgrowth(df)
    rules_path = artifacts_dir / "association_rules.parquet"
    pq.write_table(pa.Table.from_pandas(rules_df), str(rules_path))

    logger.info("Standardizing %d features for clustering & anomalies ...", len(CLUSTER_FEATURES))
    x_scaled = _prepare_feature_matrix(df)

    logger.info("Running KMeans (k=3..8) ...")
    clusters_df, best_k, best_sil = _run_kmeans(df, x_scaled)
    clusters_path = artifacts_dir / "clusters.parquet"
    pq.write_table(pa.Table.from_pandas(clusters_df), str(clusters_path))

    logger.info("Running IsolationForest ...")
    anomalies_df = _run_isolation_forest(df, x_scaled)
    anomalies_path = artifacts_dir / "anomalies.parquet"
    pq.write_table(pa.Table.from_pandas(anomalies_df), str(anomalies_path))

    logger.info("Aggregating per-film stats ...")
    film_stats_df = _build_film_stats(df)
    film_stats_path = artifacts_dir / "film_stats.parquet"
    pq.write_table(pa.Table.from_pandas(film_stats_df), str(film_stats_path))

    top_lift = float(rules_df["lift"].iloc[0]) if len(rules_df) > 0 else 0.0
    print(
        f"run_mining complete: rules={len(rules_df)} (top_lift={top_lift:.3f}), "
        f"best_k={best_k}, silhouette={best_sil:.4f}, "
        f"anomalies={len(anomalies_df)}, films={len(film_stats_df)}"
    )
    print("Note: Association is not causation.")
    if len(rules_df) > 0:
        for idx, row in rules_df.head(5).iterrows():
            print(
                f"  rule[{idx}] {row['antecedents']} -> {row['consequents']} "
                f"(sup={row['support']:.3f}, conf={row['confidence']:.3f}, lift={row['lift']:.3f})"
            )

    return {
        "n_rules": len(rules_df),
        "top_lift": top_lift,
        "best_k": best_k,
        "silhouette": best_sil,
        "n_anomalies": len(anomalies_df),
        "n_films": len(film_stats_df),
    }

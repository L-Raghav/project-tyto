"""Smoke tests for Project Tyto MVP Stage 6 predictor components."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _make_synthetic_predictor_df(n: int = 20) -> pd.DataFrame:
    """Build a 20-row synthetic source-only features DataFrame with y_flag."""
    rng = np.random.default_rng(7)
    y_flag = np.array([1 if i < 4 else 0 for i in range(n)], dtype=int)
    return pd.DataFrame(
        {
            "en_words": rng.integers(1, 20, size=n),
            "en_exclaim": rng.integers(0, 2, size=n),
            "en_question": rng.integers(0, 2, size=n),
            "en_caps_ratio": rng.uniform(0.0, 0.3, size=n),
            "digit_count": rng.integers(0, 3, size=n),
            "cap_name_count": rng.integers(0, 3, size=n),
            "idiom_present": rng.integers(0, 2, size=n),
            "repeat_count": rng.integers(1, 4, size=n),
            "is_short": rng.integers(0, 2, size=n),
            "emotion_cue_count": rng.integers(0, 3, size=n),
            "y_flag": y_flag,
        }
    )


def _stratified_bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = 200,
    seed: int = 42,
) -> list[float]:
    """Compute a stratified bootstrap 95% CI [lo, hi] for AUROC."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    scores: list[float] = []
    for _ in range(n_resamples):
        p_sample = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        n_sample = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([p_sample, n_sample])
        scores.append(float(roc_auc_score(y_true[idx], y_prob[idx])))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return [float(lo), float(hi)]


def test_predictor_smoke() -> None:
    """LogisticRegression runs on synthetic source-only features without error."""
    df = _make_synthetic_predictor_df(20)
    feature_cols = [c for c in df.columns if c != "y_flag"]
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["y_flag"].to_numpy(dtype=int)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(x, y)
    probs = clf.predict_proba(x)[:, 1]
    preds = clf.predict(x)

    assert probs.shape == (20,)
    assert preds.shape == (20,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_pca_en_embedding() -> None:
    """PCA(n=min(20, features)) works cleanly on small synthetic embedding matrices."""
    rng = np.random.default_rng(123)
    # Case 1: embedding dim (12) < 20, n_samples=25
    emb_small = rng.normal(size=(25, 12)).astype(np.float32)
    n_comp_small = min(20, emb_small.shape[0], emb_small.shape[1])
    pca_small = PCA(n_components=n_comp_small, random_state=42)
    out_small = pca_small.fit_transform(emb_small)
    assert out_small.shape == (25, 12)

    # Case 2: embedding dim (64) > 20, n_samples=25
    emb_wide = rng.normal(size=(25, 64)).astype(np.float32)
    n_comp_wide = min(20, emb_wide.shape[0], emb_wide.shape[1])
    pca_wide = PCA(n_components=n_comp_wide, random_state=42)
    out_wide = pca_wide.fit_transform(emb_wide)
    assert out_wide.shape == (25, 20)


def test_bootstrap_ci() -> None:
    """Bootstrap CI returns [lo, hi] with 0 <= lo <= hi <= 1."""
    df = _make_synthetic_predictor_df(20)
    feature_cols = [c for c in df.columns if c != "y_flag"]
    x = df[feature_cols].to_numpy(dtype=float)
    y = df["y_flag"].to_numpy(dtype=int)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(x, y)
    probs = clf.predict_proba(x)[:, 1]

    ci = _stratified_bootstrap_auroc_ci(y, probs, n_resamples=200, seed=42)
    assert len(ci) == 2
    lo, hi = ci
    assert 0.0 <= lo <= hi <= 1.0

"""Source-only heuristic-risk predictor for Project Tyto MVP (Stage 6).

Trains Baseline (median en_words), LogisticRegression, and LightGBM models
using GroupKFold cross-validation grouped by film/series on English-only
features to predict the top 20% of heuristic risk_score.

NOTE: This model predicts a heuristic proxy trained on weak labels.
It is NOT validated against human translation-quality judgement.
"""

from __future__ import annotations

import gc
import json
import logging
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tyto.mvp.features import EN_IDIOMS, _stem_phrase

logger = logging.getLogger(__name__)

# ILLUSTRATIVE — to be reviewed by a native speaker
EMOTION_CUES: list[str] = [
    "love",
    "hate",
    "angry",
    "sad",
    "happy",
    "fear",
    "shock",
    "cry",
    "laugh",
    "please",
    "sorry",
    "help",
    "kill",
    "die",
    "alive",
]
_EMOTION_CUE_SET: frozenset[str] = frozenset(EMOTION_CUES)
_WORD_ALPHA_RE = re.compile(r"[a-z]+")
_STRIP_PUNCT = ".,!?;:\"'()[]{}-—–"

BASE_SOURCE_FEATURES: list[str] = [
    "en_words",
    "en_exclaim",
    "en_question",
    "en_caps_ratio",
    "digit_count",
    "cap_name_count",
    "idiom_present",
    "repeat_count",
    "is_short",
    "emotion_cue_count",
]

PREDICTOR_NOTE = (
    "Model predicts a heuristic proxy trained on weak labels. "
    "NOT validated against human judgement."
)


def _count_mid_sentence_cap_names(text: str) -> int:
    """Count Title Case words not at sentence start as a proxy for named entities."""
    tokens = text.split()
    count = 0
    at_sentence_start = True
    for raw_tok in tokens:
        cleaned = raw_tok.strip(_STRIP_PUNCT)
        if cleaned and not at_sentence_start:
            # Exclude single-letter words such as "I" and all-caps acronyms.
            if (
                len(cleaned) > 1
                and cleaned[0].isupper()
                and any(ch.islower() for ch in cleaned[1:])
            ):
                count += 1
        end_trimmed = raw_tok.rstrip("\"'”’)]}")
        at_sentence_start = end_trimmed.endswith((".", "!", "?"))
    return count


def compute_source_features(en_texts: list[str]) -> pd.DataFrame:
    """Compute the 10 English source-only features (no Tamil text used)."""
    idioms_lower = [idiom.lower() for idiom in EN_IDIOMS]
    idioms_stemmed = [_stem_phrase(idiom) for idiom in EN_IDIOMS]

    rows: list[dict[str, float | int]] = []
    for en in en_texts:
        text = "" if en is None else str(en)
        words = text.split()
        n_words = len(words)
        en_denom = max(len(text), 1)
        caps_count = sum(1 for ch in text if ch.isupper())
        digit_count = sum(1 for ch in text if ch.isdigit())
        cap_name_count = _count_mid_sentence_cap_names(text)

        if words:
            counts = Counter(w.lower() for w in words)
            rep = counts.most_common(1)[0][1]
        else:
            rep = 0

        en_lower = text.lower()
        en_stemmed = _stem_phrase(en_lower)
        idiom_present = 0
        for idiom, idiom_stem in zip(idioms_lower, idioms_stemmed, strict=True):
            if idiom in en_lower or (idiom_stem and idiom_stem in en_stemmed):
                idiom_present = 1
                break

        alpha_tokens = _WORD_ALPHA_RE.findall(en_lower)
        emotion_cue_count = sum(1 for tok in alpha_tokens if tok in _EMOTION_CUE_SET)

        rows.append(
            {
                "en_words": n_words,
                "en_exclaim": 1 if "!" in text else 0,
                "en_question": 1 if "?" in text else 0,
                "en_caps_ratio": caps_count / en_denom,
                "digit_count": digit_count,
                "cap_name_count": cap_name_count,
                "idiom_present": idiom_present,
                "repeat_count": rep,
                "is_short": 1 if n_words <= 3 else 0,
                "emotion_cue_count": emotion_cue_count,
            }
        )

    return pd.DataFrame(rows, columns=BASE_SOURCE_FEATURES)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute stratified 95% bootstrap confidence interval for a binary metric."""
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(y_score, dtype=float)
    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        val = float(metric_fn(y_arr, s_arr))
        return val, val

    rng = np.random.default_rng(seed)
    stats: list[float] = []
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    for _ in range(n_resamples):
        p_samp = rng.choice(pos_idx, size=n_pos, replace=True)
        n_samp = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([p_samp, n_samp])
        stats.append(float(metric_fn(y_arr[idx], s_arr[idx])))

    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    frac: float = 0.10,
    seed: int = 42,
) -> float:
    """Share of true positives captured in the top ``frac`` of predicted risk."""
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(y_score, dtype=float)
    total_pos = int(y_arr.sum())
    if total_pos == 0 or len(y_arr) == 0:
        return 0.0

    k = max(1, int(np.ceil(frac * len(y_arr))))
    # Deterministic random tie-breaker so binary baselines have no row-order bias.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_arr))
    ordered = perm[np.argsort(-s_arr[perm], kind="mergesort")]
    top_idx = ordered[:k]
    return float(y_arr[top_idx].sum() / total_pos)


def _evaluate_oof(
    y_true: np.ndarray,
    oof_scores: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute AUROC, PR-AUC, stratified bootstrap 95% CIs, and recall@10%."""
    auroc = float(roc_auc_score(y_true, oof_scores))
    pr_auc = float(average_precision_score(y_true, oof_scores))
    auroc_lo, auroc_hi = bootstrap_ci(
        y_true, oof_scores, roc_auc_score, n_resamples=n_resamples, seed=seed
    )
    pr_lo, pr_hi = bootstrap_ci(
        y_true, oof_scores, average_precision_score, n_resamples=n_resamples, seed=seed
    )
    rec10 = recall_at_k(y_true, oof_scores, frac=0.10, seed=seed)
    return {
        "auroc": round(auroc, 6),
        "pr_auc": round(pr_auc, 6),
        "auroc_ci": [round(auroc_lo, 6), round(auroc_hi, 6)],
        "pr_auc_ci": [round(pr_lo, 6), round(pr_hi, 6)],
        "auroc_ci_lo": round(auroc_lo, 6),
        "auroc_ci_hi": round(auroc_hi, 6),
        "pr_auc_ci_lo": round(pr_lo, 6),
        "pr_auc_ci_hi": round(pr_hi, 6),
        "recall_at_10": round(rec10, 6),
    }


def _load_en_embeddings(artifacts_dir: Path, n_rows: int) -> np.ndarray | None:
    """Load precomputed English LaBSE embeddings if present and row-aligned."""
    emb_path = artifacts_dir / "labse_en_embs.npy"
    if not emb_path.exists():
        logger.warning(
            "Precomputed English LaBSE embeddings not found at %s; using zeros for PCA(20).",
            emb_path,
        )
        return None
    try:
        embs = np.load(emb_path)
        if embs.shape[0] != n_rows:
            logger.warning(
                "LaBSE embedding row count (%d) != dataset rows (%d); using zeros for PCA(20).",
                embs.shape[0],
                n_rows,
            )
            return None
        logger.info("Loaded English LaBSE embeddings %s from %s", embs.shape, emb_path)
        return np.asarray(embs, dtype=np.float32)
    except Exception as exc:
        logger.warning("Failed to load %s (%s); using zeros for PCA(20).", emb_path, exc)
        return None


def _encode_en_texts_labse(texts: list[str], embedder: object | None = None) -> np.ndarray | None:
    """Encode English sentences with LaBSE on GPU/CPU; returns None on failure."""
    owns_model = embedder is None
    model = embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if model is None:
            from sentence_transformers import SentenceTransformer

            try:
                model = SentenceTransformer(
                    "sentence-transformers/LaBSE", device=device, local_files_only=True
                )
            except Exception:
                model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
            if device == "cuda":
                model.half()

        embs = model.encode(  # type: ignore[attr-defined]
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        arr = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)
    except Exception as exc:
        logger.warning("LaBSE encoding unavailable for live texts (%s); using zero PCA.", exc)
        return None
    finally:
        if owns_model and model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _tier_from_prob(prob: float, p70: float, p90: float) -> str:
    """Map predicted probability to High / Medium / Low using OOF quantile cutoffs."""
    if prob >= p90:
        return "High"
    if prob >= p70:
        return "Medium"
    return "Low"


def predict_texts(
    texts: list[str],
    model_path: str | Path = "artifacts/lgbm_model.pkl",
    use_labse: bool = True,
    embedder: object | None = None,
) -> list[dict[str, Any]]:
    """Run the saved LightGBM predictor on raw English texts and return SHAP-style explanations."""
    bundle = joblib.load(Path(model_path))
    model: LGBMClassifier = bundle["model"]
    pca: PCA | None = bundle.get("pca")
    n_pca: int = int(bundle.get("n_pca_components", 20))
    feature_names: list[str] = list(bundle["feature_names"])
    prob_p70: float = float(bundle.get("prob_p70", 0.40))
    prob_p90: float = float(bundle.get("prob_p90", 0.70))

    base_df = compute_source_features(texts)
    pca_cols = [f"pca_{i}" for i in range(n_pca)]

    pca_mat = np.zeros((len(texts), n_pca), dtype=np.float32)
    if pca is not None and use_labse:
        embs = _encode_en_texts_labse(texts, embedder=embedder)
        if embs is not None:
            pca_mat = pca.transform(embs).astype(np.float32)
        else:
            logger.warning("Using zero PCA features for live prediction.")
    elif pca is not None and not use_labse:
        logger.warning("use_labse=False; using zero PCA features for live prediction.")

    pca_df = pd.DataFrame(pca_mat, columns=pca_cols)
    x_df = pd.concat([base_df.reset_index(drop=True), pca_df], axis=1)[feature_names]

    probs = model.predict_proba(x_df)[:, 1]
    contribs = model.booster_.predict(x_df, pred_contrib=True)[:, :-1]

    results: list[dict[str, Any]] = []
    for idx, text in enumerate(texts):
        prob = float(probs[idx])
        row_contrib = contribs[idx]
        top_order = np.argsort(-np.abs(row_contrib))[:8]
        top_features = [
            {
                "feature": feature_names[f_idx],
                "contribution": round(float(row_contrib[f_idx]), 6),
                "value": round(float(x_df.iloc[idx, f_idx]), 6),
            }
            for f_idx in top_order
        ]
        results.append(
            {
                "text": text,
                "risk_prob": round(prob, 6),
                "risk_tier": _tier_from_prob(prob, prob_p70, prob_p90),
                "top_features": top_features,
            }
        )
    return results


def _run_demo_predictions(
    demo_examples_path: Path,
    model_path: Path,
    out_path: Path,
) -> list[dict[str, Any]]:
    """Run the saved model on the 12 illustrative demo sentences and write JSON."""
    if not demo_examples_path.exists():
        logger.warning(
            "Demo examples file not found at %s; skipping demo predictions.",
            demo_examples_path,
        )
        return []

    examples = json.loads(demo_examples_path.read_text(encoding="utf-8"))
    texts = [str(item["text"]) for item in examples]
    preds = predict_texts(texts, model_path=model_path, use_labse=True)

    output_records: list[dict[str, Any]] = []
    for item, pred in zip(examples, preds, strict=True):
        output_records.append(
            {
                "id": item["id"],
                "type": item["type"],
                "text": item["text"],
                "label": item.get("label", "ILLUSTRATIVE inputs — not from real subtitle data"),
                "risk_tier": pred["risk_tier"],
                "risk_prob": pred["risk_prob"],
                "top_features": pred["top_features"],
            }
        )

    out_path.write_text(json.dumps(output_records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote %d demo predictions to %s", len(output_records), out_path)
    return output_records


def train_predictor(
    risk_path: str | Path = "artifacts/risk.parquet",
    artifacts_dir: str | Path = "artifacts",
    demo_examples_path: str | Path = "data/resources/demo_examples.json",
) -> dict[str, Any]:
    """Train source-only predictors with GroupKFold CV and save model, metrics, and SHAP."""
    risk_path = Path(risk_path)
    artifacts_dir = Path(artifacts_dir)
    demo_examples_path = Path(demo_examples_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading risk data from %s", risk_path)
    df = pd.read_parquet(risk_path)

    # Target: top 20% of heuristic risk_score -> y_flag = 1
    p80 = float(np.percentile(df["risk_score"].astype(float), 80))
    y = (df["risk_score"].astype(float) >= p80).astype(int).to_numpy()

    # Groups: series_group if non-empty else film_id
    series_col = df["series_group"].fillna("").astype(str)
    film_col = df["film_id"].fillna("").astype(str)
    groups = np.where(series_col != "", series_col, film_col)
    n_groups = int(pd.Series(groups).nunique())
    n_splits = min(5, n_groups)

    logger.info("Computing source-only features (English text only, %d rows)", len(df))
    en_texts = df["en"].fillna("").astype(str).tolist()
    x_base_df = compute_source_features(en_texts)
    en_words_arr = x_base_df["en_words"].to_numpy(dtype=float)

    embs = _load_en_embeddings(artifacts_dir, len(df))
    n_pca = 20 if embs is None else min(20, embs.shape[0], embs.shape[1])
    pca_cols = [f"pca_{i}" for i in range(n_pca)]
    feature_names = BASE_SOURCE_FEATURES + pca_cols

    gkf = GroupKFold(n_splits=n_splits)
    oof_baseline = np.zeros(len(df), dtype=float)
    oof_logistic = np.zeros(len(df), dtype=float)
    oof_lgbm = np.zeros(len(df), dtype=float)

    logger.info("Running %d-fold GroupKFold across %d groups ...", n_splits, n_groups)
    x_base_arr = x_base_df.to_numpy(dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(x_base_arr, y, groups)):
        # 1. Word-count-only baseline
        train_med = float(np.median(en_words_arr[train_idx]))
        oof_baseline[val_idx] = (en_words_arr[val_idx] > train_med).astype(float)

        # Leak-free PCA(20) inside each fold
        if embs is not None:
            fold_pca = PCA(n_components=n_pca, random_state=42)
            pca_train = fold_pca.fit_transform(embs[train_idx]).astype(np.float32)
            pca_val = fold_pca.transform(embs[val_idx]).astype(np.float32)
        else:
            pca_train = np.zeros((len(train_idx), n_pca), dtype=np.float32)
            pca_val = np.zeros((len(val_idx), n_pca), dtype=np.float32)

        x_train = pd.DataFrame(
            np.hstack([x_base_arr[train_idx], pca_train]), columns=feature_names
        )
        x_val = pd.DataFrame(
            np.hstack([x_base_arr[val_idx], pca_val]), columns=feature_names
        )

        # 2. LogisticRegression
        logreg = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        logreg.fit(x_train, y[train_idx])
        oof_logistic[val_idx] = logreg.predict_proba(x_val)[:, 1]

        # 3. LightGBM
        lgbm = LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(x_train, y[train_idx])
        oof_lgbm[val_idx] = lgbm.predict_proba(x_val)[:, 1]
        logger.info("Fold %d/%d complete (val=%d)", fold_idx + 1, n_splits, len(val_idx))

    logger.info("Computing pooled OOF metrics + 1000-resample stratified bootstrap CIs ...")
    metrics: dict[str, Any] = {
        "baseline": _evaluate_oof(y, oof_baseline),
        "logistic": _evaluate_oof(y, oof_logistic),
        "lgbm": _evaluate_oof(y, oof_lgbm),
        "n_samples": int(len(df)),
        "n_positives": int(y.sum()),
        "positive_rate": round(float(y.mean()), 6),
        "n_folds": n_splits,
        "n_groups": n_groups,
        "p80_risk_threshold": round(p80, 6),
        "note": PREDICTOR_NOTE,
    }
    metrics_path = artifacts_dir / "predictor_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Fit final LightGBM + PCA on full dataset for deployment & SHAP feature importance
    logger.info("Fitting final LightGBM model on full dataset (%d rows) ...", len(df))
    if embs is not None:
        final_pca: PCA | None = PCA(n_components=n_pca, random_state=42)
        pca_full = final_pca.fit_transform(embs).astype(np.float32)
    else:
        final_pca = None
        pca_full = np.zeros((len(df), n_pca), dtype=np.float32)

    x_full = pd.DataFrame(np.hstack([x_base_arr, pca_full]), columns=feature_names)
    final_lgbm = LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final_lgbm.fit(x_full, y)

    prob_p70 = float(np.percentile(oof_lgbm, 70))
    prob_p90 = float(np.percentile(oof_lgbm, 90))
    model_path = artifacts_dir / "lgbm_model.pkl"
    bundle = {
        "model": final_lgbm,
        "pca": final_pca,
        "feature_names": feature_names,
        "n_pca_components": n_pca,
        "prob_p70": prob_p70,
        "prob_p90": prob_p90,
        "note": PREDICTOR_NOTE,
    }
    joblib.dump(bundle, model_path)
    logger.info("Saved LightGBM bundle to %s", model_path)

    # SHAP-style feature importance via LightGBM pred_contrib
    contribs = final_lgbm.booster_.predict(x_full, pred_contrib=True)[:, :-1]
    mean_abs = np.mean(np.abs(contribs), axis=0)
    mean_signed = np.mean(contribs, axis=0)
    order = np.argsort(-mean_abs)
    importance_list = [
        {
            "feature": feature_names[idx],
            "mean_abs_contrib": round(float(mean_abs[idx]), 6),
            "mean_contrib": round(float(mean_signed[idx]), 6),
        }
        for idx in order
    ]
    fi_payload = {
        "method": "lightgbm_pred_contrib_mean_abs",
        "note": PREDICTOR_NOTE,
        "features": importance_list,
    }
    fi_path = artifacts_dir / "feature_importance.json"
    fi_path.write_text(json.dumps(fi_payload, indent=2), encoding="utf-8")

    # Run real model on 12 illustrative demo examples
    demo_out_path = artifacts_dir / "demo_predictions.json"
    _run_demo_predictions(demo_examples_path, model_path, demo_out_path)

    lgbm_m = metrics["lgbm"]
    log_m = metrics["logistic"]
    base_m = metrics["baseline"]
    print(
        f"train_predictor complete ({len(df):,} rows, pos={int(y.sum()):,} [{y.mean():.1%}]):\n"
        f"  Baseline : AUROC={base_m['auroc']:.4f} {base_m['auroc_ci']}, "
        f"PR-AUC={base_m['pr_auc']:.4f} {base_m['pr_auc_ci']}, R@10%={base_m['recall_at_10']:.4f}\n"
        f"  LogReg   : AUROC={log_m['auroc']:.4f} {log_m['auroc_ci']}, "
        f"PR-AUC={log_m['pr_auc']:.4f} {log_m['pr_auc_ci']}, R@10%={log_m['recall_at_10']:.4f}\n"
        f"  LightGBM : AUROC={lgbm_m['auroc']:.4f} {lgbm_m['auroc_ci']}, "
        f"PR-AUC={lgbm_m['pr_auc']:.4f} {lgbm_m['pr_auc_ci']}, R@10%={lgbm_m['recall_at_10']:.4f}\n"
        f"  Top features: {', '.join(f['feature'] for f in importance_list[:5])}\n"
        f"  Note: {PREDICTOR_NOTE}"
    )
    return metrics

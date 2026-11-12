"""QC filtering pipeline: applies per-row flags and computes dedup statistics.

Stages applied to the pairs DataFrame:

1. **Length filters**: min/max token counts for en and ta, max character
   length ratio.
2. **Script-based language ID**: Tamil Unicode block ratio, ASCII ratio;
   flags WRONG_LANG, ROMANISED_TA, CODE_MIXED.
3. **Exact duplicate detection**: identical (en, ta) pairs get a
   ``repeat_count`` column; flag ``EXACT_DUP`` on count > 1.
4. **Near-duplicate detection** (MinHash LSH): Jaccard-similar pairs get
   flag ``NEAR_DUP``.  We keep all rows but set the flag.
5. **LaBSE alignment-score flag** (stage 2, optional): rows below
   ``labse_min_score`` get flag ``LOW_LABSE``.

Flags are appended to the ``qc_flags`` column (pipe-separated).
A row is **usable** if it has no qc_flags that disqualify it:
    WRONG_LANG, TOO_SHORT, TOO_LONG, BAD_RATIO  (hard failures)
    EXACT_DUP, NEAR_DUP, ROMANISED_TA, CODE_MIXED, LOW_LABSE (soft warnings)

Stop-gate: if usable pair count < cfg.qc.min_usable_pairs, raises RuntimeError.

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tyto.utils.config import TytoConfig

logger = logging.getLogger(__name__)

# Flags that make a pair hard-fail (excluded from usable count).
HARD_FAIL_FLAGS = frozenset({"WRONG_LANG", "TOO_SHORT_EN", "TOO_SHORT_TA",
                              "TOO_LONG_EN", "TOO_LONG_TA", "BAD_RATIO"})


def _add_flag(series: pd.Series, flag: str, mask: pd.Series) -> pd.Series:
    """Append *flag* to the qc_flags cell for every row where *mask* is True."""
    updated = series.copy()
    has_flag = updated[mask]
    updated[mask] = has_flag.apply(
        lambda v: f"{v}|{flag}" if v else flag
    )
    return updated


def apply_length_filters(df: pd.DataFrame, cfg: TytoConfig) -> pd.DataFrame:
    """Flag rows that violate length bounds or length ratio.

    Flags added: ``TOO_SHORT_EN``, ``TOO_SHORT_TA``, ``TOO_LONG_EN``,
    ``TOO_LONG_TA``, ``BAD_RATIO``.

    Args:
        df:  DataFrame with ``en`` and ``ta`` columns.
        cfg: TytoConfig with QC thresholds.

    Returns:
        DataFrame with ``qc_flags`` column updated in-place copy.
    """
    qc = cfg.qc
    df = df.copy()

    en_tok = df["en"].fillna("").str.split().str.len()
    ta_tok = df["ta"].fillna("").str.split().str.len()

    df["qc_flags"] = _add_flag(df["qc_flags"], "TOO_SHORT_EN", en_tok < qc.min_en_tokens)
    df["qc_flags"] = _add_flag(df["qc_flags"], "TOO_SHORT_TA", ta_tok < qc.min_ta_tokens)
    df["qc_flags"] = _add_flag(df["qc_flags"], "TOO_LONG_EN", en_tok > qc.max_en_tokens)
    df["qc_flags"] = _add_flag(df["qc_flags"], "TOO_LONG_TA", ta_tok > qc.max_ta_tokens)

    # Length ratio using character lengths.
    en_len = df["en"].fillna("").str.len().clip(lower=1)
    ta_len = df["ta"].fillna("").str.len().clip(lower=1)
    ratio = np.maximum(en_len.values, ta_len.values) / np.minimum(en_len.values, ta_len.values)
    df["qc_flags"] = _add_flag(
        df["qc_flags"],
        "BAD_RATIO",
        pd.Series(ratio > qc.max_length_ratio, index=df.index),
    )

    flagged = (df["qc_flags"] != "").sum()
    logger.info("Length filter: %d rows flagged", flagged)
    return df


def apply_script_flags(df: pd.DataFrame, cfg: TytoConfig) -> pd.DataFrame:
    """Apply script-based language-ID flags to the Tamil column.

    Flags added: ``WRONG_LANG``, ``ROMANISED_TA``, ``CODE_MIXED``.

    Args:
        df:  DataFrame with ``ta`` column.
        cfg: TytoConfig with QC thresholds.

    Returns:
        DataFrame with ``qc_flags`` updated.
    """
    from tyto.data.clean import compute_script_flags

    qc = cfg.qc
    df = df.copy()

    new_flags = df["ta"].fillna("").apply(
        lambda t: compute_script_flags(
            t,
            tamil_ratio_min=qc.tamil_ratio_min,
            tamil_ratio_code_mix=qc.tamil_ratio_code_mix,
            ascii_ratio_romanised=qc.ascii_ratio_romanised,
        )
    )

    for flag in ["WRONG_LANG", "ROMANISED_TA", "CODE_MIXED"]:
        mask = new_flags.apply(lambda flags, f=flag: f in flags)
        df["qc_flags"] = _add_flag(df["qc_flags"], flag, mask)

    flagged = new_flags.apply(bool).sum()
    logger.info("Script-ID flags: %d rows flagged", flagged)
    return df


def apply_exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Count exact duplicate (en, ta) pairs and flag duplicates.

    A new column ``repeat_count`` (int) is added: 1 = unique, >1 = duplicate.
    Rows with repeat_count > 1 receive flag ``EXACT_DUP``.

    Args:
        df: DataFrame with ``en`` and ``ta`` columns.

    Returns:
        DataFrame with ``repeat_count`` column added and ``qc_flags`` updated.
    """
    df = df.copy()
    counts = df.groupby(["en", "ta"], sort=False).transform("size")
    df["repeat_count"] = counts.astype("int32")
    dup_mask = df["repeat_count"] > 1
    df["qc_flags"] = _add_flag(df["qc_flags"], "EXACT_DUP", dup_mask)
    logger.info(
        "Exact dedup: %d unique groups, %d duplicates",
        (df["repeat_count"] == 1).sum(),
        dup_mask.sum(),
    )
    return df


def apply_near_dedup(df: pd.DataFrame, cfg: TytoConfig) -> pd.DataFrame:
    """Flag near-duplicate pairs using MinHash LSH on the English text.

    Rows identified as near-duplicates receive flag ``NEAR_DUP``.
    All rows are kept; only the flag is set.

    MinHash is computed over 4-character shingles of the lowercased English
    sentence.

    Args:
        df:  DataFrame with ``en`` column.
        cfg: TytoConfig (minhash_num_perm, minhash_threshold).

    Returns:
        DataFrame with ``qc_flags`` updated.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning("datasketch not installed – skipping near-dedup (flag NEAR_DUP).")
        return df

    df = df.copy()
    qc = cfg.qc
    num_perm = qc.minhash_num_perm
    threshold = qc.minhash_threshold

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, MinHash] = {}

    def _shingles(text: str, k: int = 4) -> set[str]:
        t = text.lower()
        return {t[i : i + k] for i in range(max(1, len(t) - k + 1))}

    near_dup_indices: set[int] = set()

    for idx, row in df.iterrows():
        text = str(row["en"]) if pd.notna(row["en"]) else ""
        m = MinHash(num_perm=num_perm)
        for shingle in _shingles(text):
            m.update(shingle.encode("utf-8"))

        try:
            result = lsh.query(m)
            if result:
                near_dup_indices.add(idx)  # type: ignore[arg-type]
                for other_idx in result:
                    near_dup_indices.add(other_idx)
        except Exception:
            pass

        try:
            lsh.insert(idx, m)
            minhashes[idx] = m  # type: ignore[index]
        except Exception:
            pass

    if near_dup_indices:
        near_dup_mask = df.index.isin(near_dup_indices)
        df["qc_flags"] = _add_flag(
            df["qc_flags"], "NEAR_DUP", pd.Series(near_dup_mask, index=df.index)
        )
        logger.info("Near-dedup: %d rows flagged as NEAR_DUP", len(near_dup_indices))
    else:
        logger.info("Near-dedup: 0 near-duplicate groups found")

    return df


def apply_labse_flags(
    df: pd.DataFrame,
    cfg: TytoConfig,
    registry: object | None = None,
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Flag pairs with low LaBSE cosine similarity as probable misalignments.

    This is a **stage-2** flag only – it requires the GPU / LaBSE model.
    If the model is unavailable the function logs a warning and returns *df*
    unchanged.

    Flag added: ``LOW_LABSE``.  A new column ``labse_score`` (float32) is also
    written to the DataFrame.

    Args:
        df:         DataFrame with ``en`` and ``ta`` columns.
        cfg:        TytoConfig.
        registry:   ModelRegistry instance (defaults to default_registry).
        batch_size: Embedding batch size.

    Returns:
        DataFrame with ``labse_score`` column and ``qc_flags`` updated.
    """
    import torch

    if registry is None:
        from tyto.modeling.registry import default_registry as registry  # type: ignore[assignment]

    df = df.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = registry.get("labse", device=device, fp16=(device == "cuda"))  # type: ignore[union-attr]
    except Exception as exc:
        logger.warning("LaBSE unavailable – skipping LOW_LABSE flag: %s", exc)
        df["labse_score"] = pd.array([None] * len(df), dtype=pd.Float32Dtype())
        return df

    logger.info("Computing LaBSE scores for %d pairs on %s …", len(df), device)

    en_texts = df["en"].fillna("").tolist()
    ta_texts = df["ta"].fillna("").tolist()

    def _encode_batch(texts: list[str]) -> np.ndarray:
        all_embs: list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_embs.append(emb)
        return np.vstack(all_embs)

    import numpy as np

    en_embs = _encode_batch(en_texts)
    ta_embs = _encode_batch(ta_texts)

    scores = (en_embs * ta_embs).sum(axis=1).astype("float32")  # dot of normalised = cosine

    df["labse_score"] = scores
    low_mask = pd.Series(scores < cfg.qc.labse_min_score, index=df.index)
    df["qc_flags"] = _add_flag(df["qc_flags"], "LOW_LABSE", low_mask)
    logger.info("LaBSE flags: %d rows below threshold %.2f", low_mask.sum(), cfg.qc.labse_min_score)
    return df


def count_usable(df: pd.DataFrame) -> int:
    """Return number of rows with no hard-fail QC flags.

    Args:
        df: DataFrame with ``qc_flags`` column.

    Returns:
        Count of rows whose qc_flags contain no HARD_FAIL_FLAGS.
    """
    def _is_hard_fail(flags: str) -> bool:
        return bool(HARD_FAIL_FLAGS.intersection(set(flags.split("|"))))

    return int((~df["qc_flags"].fillna("").apply(_is_hard_fail)).sum())


def run_qc_pipeline(
    df: pd.DataFrame,
    cfg: TytoConfig,
    *,
    run_labse: bool = False,
    labse_registry: object | None = None,
) -> pd.DataFrame:
    """Run the full QC pipeline (steps 1–5) on *df*.

    Args:
        df:              Input pairs DataFrame.
        cfg:             TytoConfig with all thresholds.
        run_labse:       Whether to run the LaBSE alignment-score stage.
        labse_registry:  ModelRegistry to use for LaBSE (None → default).

    Returns:
        DataFrame with ``repeat_count``, ``labse_score`` (if run_labse),
        and updated ``qc_flags``.

    Raises:
        RuntimeError: If usable pair count is below cfg.qc.min_usable_pairs.
    """
    logger.info("Starting QC pipeline on %d rows …", len(df))
    df = apply_length_filters(df, cfg)
    df = apply_script_flags(df, cfg)
    df = apply_exact_dedup(df)
    df = apply_near_dedup(df, cfg)

    if run_labse:
        df = apply_labse_flags(df, cfg, registry=labse_registry)

    usable = count_usable(df)
    total = len(df)
    logger.info(
        "QC complete: %d / %d usable pairs (%.1f %%)",
        usable,
        total,
        100.0 * usable / max(total, 1),
    )

    if usable < cfg.qc.min_usable_pairs:
        raise RuntimeError(
            f"STOP GATE: only {usable:,} usable pairs remain "
            f"(threshold: {cfg.qc.min_usable_pairs:,}). "
            "Review data_audit.md for options before continuing."
        )

    return df

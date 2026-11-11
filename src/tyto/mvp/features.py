"""Feature extraction for Project Tyto MVP (Stage 3).

Computes cheap source/target surface features, illustrative English idiom
matches, illustrative Tamil hedge counts, and LaBSE bilingual cosine similarity
with GPU fp16 batching + OOM backoff and disk embedding cache.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

logger = logging.getLogger(__name__)

# ILLUSTRATIVE — to be reviewed by a native speaker
TA_HEDGE_WORDS: list[str] = [
    "ஒருவேளை",
    "ஒருவித",
    "சற்று",
    "கொஞ்சம்",
    "ஏதாவது",
]

# ILLUSTRATIVE — to be reviewed by a native speaker (40 English idioms)
EN_IDIOMS: list[str] = [
    "kick the bucket",
    "bite the bullet",
    "break a leg",
    "hit the nail",
    "spill the beans",
    "cost an arm",
    "under the weather",
    "piece of cake",
    "once in a blue moon",
    "pull someone's leg",
    "blessing in disguise",
    "beat around the bush",
    "bite the hand",
    "burn bridges",
    "cut corners",
    "devil in the details",
    "face the music",
    "get out of hand",
    "hang in there",
    "it takes two",
    "jump on the bandwagon",
    "kill two birds",
    "let the cat out",
    "miss the boat",
    "no pain no gain",
    "once bitten",
    "over the moon",
    "read between the lines",
    "see eye to eye",
    "sit on the fence",
    "steal someone's thunder",
    "the ball is in your court",
    "throw in the towel",
    "under the bus",
    "up in the air",
    "wild goose chase",
    "wolf in sheep",
    "wrap your head",
    "you can't have it both ways",
    "bite off more than you can chew",
]

_SPACED_PUNCT_PATTERNS = (" , ", " . ", " ! ", " ? ")
_PUNCT_SET = frozenset(string.punctuation)
_DIGIT_RE = re.compile(r"\d+")
_WORD_RE = re.compile(r"[a-z']+")
_VOWEL_LO = 0x0BBE
_VOWEL_HI = 0x0BCC


def _stem_word(w: str) -> str:
    """Lightweight suffix stripper so inflected verbs match idiom phrases."""
    if len(w) > 4 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 3 and w.endswith("ed"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("es"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def _stem_phrase(text: str) -> str:
    """Tokenize and stem lowercase alphabetic words into a space-joined string."""
    return " ".join(_stem_word(w) for w in _WORD_RE.findall(text.lower()))


def _count_puncts(text: str) -> int:
    """Count ASCII punctuation characters in text."""
    return sum(1 for ch in text if ch in _PUNCT_SET)


def _sha256_texts(texts: list[str]) -> str:
    """Compute SHA-256 hex digest of a list of texts."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_cheap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cheap surface and illustrative idiom/hedge features on a DataFrame.

    Expects columns ``en`` and ``ta``. Returns a new DataFrame with all
    original columns plus extracted cheap features.
    """
    out = df.copy()
    en_list = out["en"].fillna("").astype(str).tolist()
    ta_list = out["ta"].fillna("").astype(str).tolist()

    en_words_col: list[int] = []
    ta_chars_col: list[int] = []
    len_ratio_col: list[float] = []
    ta_vowel_share_col: list[float] = []
    ta_latin_share_col: list[float] = []
    ta_spaced_punct_col: list[float] = []
    en_exclaim_col: list[int] = []
    en_question_col: list[int] = []
    en_caps_ratio_col: list[float] = []
    digit_match_col: list[int] = []
    punct_parity_col: list[int] = []
    repeat_count_col: list[int] = []
    is_short_col: list[int] = []
    ta_hedge_count_col: list[int] = []
    idiom_present_col: list[int] = []
    idiom_id_col: list[int] = []

    idioms_lower = [idiom.lower() for idiom in EN_IDIOMS]
    idioms_stemmed = [_stem_phrase(idiom) for idiom in EN_IDIOMS]

    for en, ta in zip(en_list, ta_list, strict=True):
        words = en.split()
        n_en_words = len(words)
        n_ta_chars = len(ta)
        ta_denom = max(n_ta_chars, 1)
        en_denom = max(len(en), 1)

        ratio = min(n_ta_chars / (n_en_words + 1), 999.0) if n_ta_chars > 0 else 0.0

        vowel_count = sum(1 for ch in ta if _VOWEL_LO <= ord(ch) <= _VOWEL_HI)
        latin_count = sum(
            1 for ch in ta if ("a" <= ch <= "z") or ("A" <= ch <= "Z")
        )
        spaced_punct_count = sum(ta.count(pat) for pat in _SPACED_PUNCT_PATTERNS)

        caps_count = sum(1 for ch in en if ch.isupper())

        en_digits = _DIGIT_RE.findall(en)
        if en_digits:
            ta_digits = set(_DIGIT_RE.findall(ta))
            d_match = 1 if any(d in ta_digits for d in en_digits) else 0
        else:
            d_match = 0

        p_parity = 1 if abs(_count_puncts(en) - _count_puncts(ta)) > 1 else 0

        if words:
            counts = Counter(w.lower() for w in words)
            rep = counts.most_common(1)[0][1]
        else:
            rep = 0

        hedge_cnt = sum(ta.count(hw) for hw in TA_HEDGE_WORDS)

        en_lower = en.lower()
        en_stemmed = _stem_phrase(en_lower)
        matched_idx = -1
        for idx, (idiom, idiom_stem) in enumerate(
            zip(idioms_lower, idioms_stemmed, strict=True)
        ):
            if idiom in en_lower or (idiom_stem and idiom_stem in en_stemmed):
                matched_idx = idx
                break

        en_words_col.append(n_en_words)
        ta_chars_col.append(n_ta_chars)
        len_ratio_col.append(float(ratio))
        ta_vowel_share_col.append(vowel_count / ta_denom)
        ta_latin_share_col.append(latin_count / ta_denom)
        ta_spaced_punct_col.append(spaced_punct_count / ta_denom)
        en_exclaim_col.append(1 if "!" in en else 0)
        en_question_col.append(1 if "?" in en else 0)
        en_caps_ratio_col.append(caps_count / en_denom)
        digit_match_col.append(d_match)
        punct_parity_col.append(p_parity)
        repeat_count_col.append(rep)
        is_short_col.append(1 if n_en_words <= 3 else 0)
        ta_hedge_count_col.append(hedge_cnt)
        idiom_present_col.append(1 if matched_idx >= 0 else 0)
        idiom_id_col.append(matched_idx)

    out["en_words"] = en_words_col
    out["ta_chars"] = ta_chars_col
    out["len_ratio"] = len_ratio_col
    out["ta_vowel_share"] = ta_vowel_share_col
    out["ta_latin_share"] = ta_latin_share_col
    out["ta_spaced_punct"] = ta_spaced_punct_col
    out["en_exclaim"] = en_exclaim_col
    out["en_question"] = en_question_col
    out["en_caps_ratio"] = en_caps_ratio_col
    out["digit_match"] = digit_match_col
    out["punct_parity"] = punct_parity_col
    out["repeat_count"] = repeat_count_col
    out["is_short"] = is_short_col
    out["ta_hedge_count"] = ta_hedge_count_col
    out["idiom_present"] = idiom_present_col
    out["idiom_id"] = idiom_id_col
    return out


def _encode_with_backoff(
    model: object,
    texts: list[str],
    batch_sizes: tuple[int, ...] = (128, 64, 32),
) -> np.ndarray:
    """Encode texts using SentenceTransformer with CUDA OOM batch-size backoff."""
    last_err: Exception | None = None
    for bs in batch_sizes:
        try:
            embs = model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=bs,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.asarray(embs, dtype=np.float32)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                logger.warning("CUDA OOM at batch_size=%d; backing off", bs)
                torch.cuda.empty_cache()
                last_err = exc
            else:
                raise
    raise RuntimeError(f"Encoding failed at all batch sizes {batch_sizes}: {last_err}")


def _compute_labse_cosine(
    en_texts: list[str],
    ta_texts: list[str],
    artifacts_dir: Path,
) -> tuple[np.ndarray, str]:
    """Compute or load cached LaBSE embeddings and return per-pair cosine similarities.

    Returns ``(cos_array, status_str)``. On failure returns NaN array and
    status ``"failed"``.
    """
    n = len(en_texts)
    en_cache = artifacts_dir / "labse_en_embs.npy"
    ta_cache = artifacts_dir / "labse_ta_embs.npy"
    en_hash_path = artifacts_dir / "labse_en_embs.sha256"
    ta_hash_path = artifacts_dir / "labse_ta_embs.sha256"

    en_hash = _sha256_texts(en_texts)
    ta_hash = _sha256_texts(ta_texts)

    # Check valid cache
    if en_cache.exists() and ta_cache.exists():
        try:
            en_embs = np.load(en_cache)
            ta_embs = np.load(ta_cache)
            hash_ok = True
            if en_hash_path.exists() and ta_hash_path.exists():
                hash_ok = (
                    en_hash_path.read_text(encoding="utf-8").strip() == en_hash
                    and ta_hash_path.read_text(encoding="utf-8").strip() == ta_hash
                )
            if en_embs.shape[0] == n and ta_embs.shape[0] == n and hash_ok:
                logger.info("Loaded cached LaBSE embeddings (%d rows)", n)
                cos = np.sum(en_embs * ta_embs, axis=1).astype(np.float32)
                return cos, "cached"
        except Exception as exc:
            logger.warning("Failed reading LaBSE cache (%s); recomputing", exc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformers/LaBSE on %s", device)
        try:
            model = SentenceTransformer(
                "sentence-transformers/LaBSE", device=device, local_files_only=True
            )
        except Exception:
            logger.info("LaBSE not in local cache; trying online download")
            model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
        if device == "cuda":
            model.half()

        en_embs = _encode_with_backoff(model, en_texts)
        ta_embs = _encode_with_backoff(model, ta_texts)

        # Normalize explicitly in case fp16 drifted slightly
        en_norm = np.linalg.norm(en_embs, axis=1, keepdims=True)
        ta_norm = np.linalg.norm(ta_embs, axis=1, keepdims=True)
        en_embs = en_embs / np.maximum(en_norm, 1e-12)
        ta_embs = ta_embs / np.maximum(ta_norm, 1e-12)

        np.save(en_cache, en_embs)
        np.save(ta_cache, ta_embs)
        en_hash_path.write_text(en_hash, encoding="utf-8")
        ta_hash_path.write_text(ta_hash, encoding="utf-8")

        cos = np.sum(en_embs * ta_embs, axis=1).astype(np.float32)
        return cos, f"ok ({device})"
    except Exception as exc:
        logger.warning("LaBSE failed (%s); setting labse_cos = NaN", exc)
        return np.full(n, np.nan, dtype=np.float32), "failed"
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_features(
    pairs_path: str | Path,
    out_path: str | Path,
    artifacts_dir: str | Path = "artifacts",
) -> pd.DataFrame:
    """Extract cheap features + LaBSE cosine similarity and write features.parquet.

    Parameters
    ----------
    pairs_path:
        Input parquet produced by ``load_v2018``.
    out_path:
        Output parquet path (default ``artifacts/features.parquet``).
    artifacts_dir:
        Directory for cached ``labse_en_embs.npy`` / ``labse_ta_embs.npy``.

    Returns
    -------
    pd.DataFrame
        DataFrame with all original columns, cheap features, and ``labse_cos``.
    """
    pairs_path = Path(pairs_path)
    out_path = Path(out_path)
    artifacts_dir = Path(artifacts_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pairs from %s", pairs_path)
    df = pd.read_parquet(pairs_path)
    logger.info("Computing cheap + illustrative idiom/hedge features (%d rows)", len(df))
    feat_df = compute_cheap_features(df)

    en_texts = feat_df["en"].fillna("").astype(str).tolist()
    ta_texts = feat_df["ta"].fillna("").astype(str).tolist()
    labse_cos, labse_status = _compute_labse_cosine(en_texts, ta_texts, artifacts_dir)
    feat_df["labse_cos"] = labse_cos

    logger.info("Writing features to %s", out_path)
    pq.write_table(pa.Table.from_pandas(feat_df), str(out_path))

    valid_cos = feat_df["labse_cos"].dropna()
    cos_mean = f"{valid_cos.mean():.4f}" if len(valid_cos) > 0 else "NaN"
    print(
        f"extract_features complete: {len(feat_df):,} rows, "
        f"idioms={int(feat_df['idiom_present'].sum())}, "
        f"labse={labse_status}, mean_cos={cos_mean} → {out_path}"
    )
    return feat_df

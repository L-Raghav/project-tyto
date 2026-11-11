"""Unit tests for Project Tyto MVP Stage 3 cheap surface features."""

from __future__ import annotations

import pandas as pd
import pytest

from tyto.mvp.features import compute_cheap_features


def _make_df(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a minimal synthetic pairs DataFrame for feature extraction."""
    return pd.DataFrame(
        [{"pair_id": idx, "en": en, "ta": ta} for idx, (en, ta) in enumerate(pairs)]
    )


def test_en_words() -> None:
    """A 3-word English string yields en_words=3."""
    df = _make_df([("one two three", "ஒன்று இரண்டு மூன்று")])
    feat = compute_cheap_features(df)
    assert int(feat.loc[0, "en_words"]) == 3


def test_len_ratio() -> None:
    """Verify len_ratio == len(ta) / (en_words + 1), and 0.0 for empty ta."""
    en_text = "hello world"  # 2 words -> denom = 3
    ta_text = "வணக்கம்"  # len = 7
    df = _make_df([(en_text, ta_text), ("hello", "")])
    feat = compute_cheap_features(df)
    assert float(feat.loc[0, "len_ratio"]) == pytest.approx(len(ta_text) / 3.0)
    assert float(feat.loc[1, "len_ratio"]) == 0.0


def test_ta_vowel_share() -> None:
    """Inject known Tamil vowel signs (U+0BBE-U+0BCC) and verify exact ratio."""
    # "கா கி" has 5 chars: 'க', 'ா'(0BBE), ' ', 'க', 'ி'(0BBF) -> 2/5 = 0.4
    ta_text = "கா கி"
    df = _make_df([("test", ta_text)])
    feat = compute_cheap_features(df)
    assert float(feat.loc[0, "ta_vowel_share"]) == pytest.approx(2.0 / 5.0)


def test_ta_latin_share() -> None:
    """Mix ASCII letters and Tamil characters and verify ta_latin_share."""
    # "abc கா" -> 6 chars, 3 ASCII letters ('a','b','c') -> 3/6 = 0.5
    ta_text = "abc கா"
    df = _make_df([("test", ta_text)])
    feat = compute_cheap_features(df)
    assert float(feat.loc[0, "ta_latin_share"]) == pytest.approx(3.0 / 6.0)


def test_digit_match() -> None:
    """Matching numeric tokens yield digit_match=1; unmatched digits yield 0."""
    df = _make_df(
        [
            ("12 items", "12 பொருட்கள்"),
            ("12 items", "twenty items"),
        ]
    )
    feat = compute_cheap_features(df)
    assert int(feat.loc[0, "digit_match"]) == 1
    assert int(feat.loc[1, "digit_match"]) == 0


def test_idiom_present() -> None:
    """Known English idiom triggers idiom_present=1."""
    df = _make_df(
        [
            ("He kicked the bucket", "அவர் இறந்துவிட்டார்"),
            ("She walked to school", "அவள் பள்ளிக்கு நடந்தாள்"),
        ]
    )
    feat = compute_cheap_features(df)
    assert int(feat.loc[0, "idiom_present"]) == 1
    assert int(feat.loc[1, "idiom_present"]) == 0


def test_is_short() -> None:
    """2-word English yields is_short=1; 5-word English yields is_short=0."""
    df = _make_df(
        [
            ("Come here", "இங்கே வா"),
            ("We are going home now", "நாங்கள் இப்போது வீட்டிற்கு செல்கிறோம்"),
        ]
    )
    feat = compute_cheap_features(df)
    assert int(feat.loc[0, "is_short"]) == 1
    assert int(feat.loc[1, "is_short"]) == 0

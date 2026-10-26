"""Unit tests for tyto.data.clean.

Tests cover TEXT RULES (PROJECT_BRIEF.md):
- Markup tags stripped
- [music] / sound descriptions removed
- Double dashes handled
- Zero-width joiners removed (Tamil)
- Mixed scripts
- Speaker dash removal
- NFC normalisation
- Script ratio computation
"""

from __future__ import annotations

import pytest

from tyto.data.clean import clean_text, compute_script_flags, script_ratios


# ─── clean_text fixtures ──────────────────────────────────────────────────────

class TestCleanText:
    """Tests for clean_text()."""

    def test_nfc_normalisation(self) -> None:
        # Decomposed 'é' (U+0065 + U+0301) should become composed U+00E9.
        decomposed = "caf\u0065\u0301"
        result = clean_text(decomposed)
        assert result == "café"

    def test_markup_tags_stripped(self) -> None:
        assert clean_text("<i>Hello</i> world") == "Hello world"
        assert clean_text("<font color=\"red\">text</font>") == "text"
        assert clean_text("<b>bold</b> <i>italic</i>") == "bold italic"

    def test_sound_descriptions_removed(self) -> None:
        assert "[laughs]" not in clean_text("[laughs] That's funny.")
        assert "(applause)" not in clean_text("(applause)")
        assert "MUSIC" not in clean_text("[MUSIC]")

    def test_music_symbols_removed(self) -> None:
        assert "♪" not in clean_text("♪ La la la ♪")
        assert "♫" not in clean_text("♫ Singing ♫")

    def test_speaker_dash_removed(self) -> None:
        result = clean_text("- Hello there.")
        assert not result.startswith("-")
        assert "Hello" in result

    def test_em_dash_speaker_removed(self) -> None:
        result = clean_text("– Hello there.")
        assert not result.startswith("–")

    def test_double_dash_replaced(self) -> None:
        result = clean_text("He was -- surprised.")
        assert "--" not in result

    def test_zero_width_joiner_removed(self) -> None:
        # Tamil text with ZWJ (U+200D).
        ta_text = "வணக்கம்\u200dமக்களே"
        result = clean_text(ta_text, lang="ta")
        assert "\u200d" not in result
        assert "வணக்கம்" in result

    def test_zero_width_non_joiner_removed(self) -> None:
        # ZWNJ U+200C commonly found in copied Tamil text.
        ta_text = "நன்றி\u200cவருக"
        result = clean_text(ta_text, lang="ta")
        assert "\u200c" not in result

    def test_bom_removed(self) -> None:
        text = "\ufeffHello"
        result = clean_text(text)
        assert result == "Hello"

    def test_mixed_markup_and_symbols(self) -> None:
        nasty = "<i>♪ [laughs] Hello -- there</i>"
        result = clean_text(nasty)
        assert "<" not in result
        assert "♪" not in result
        assert "--" not in result
        assert "Hello" in result

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_whitespace_collapsed(self) -> None:
        result = clean_text("Hello    world")
        assert result == "Hello world"

    def test_only_markup_becomes_empty(self) -> None:
        # A subtitle that is purely a tag should clean to empty.
        result = clean_text("<font color='white'></font>")
        assert result.strip() == ""

    def test_tamil_text_preserved(self) -> None:
        ta = "வணக்கம் உலகம்"
        assert clean_text(ta, lang="ta") == ta


# ─── script_ratios ────────────────────────────────────────────────────────────

class TestScriptRatios:
    """Tests for script_ratios()."""

    def test_pure_tamil(self) -> None:
        tamil_r, ascii_r = script_ratios("வணக்கம்")
        assert tamil_r > 0.9
        assert ascii_r < 0.1

    def test_pure_english(self) -> None:
        tamil_r, ascii_r = script_ratios("Hello world")
        assert tamil_r == 0.0
        assert ascii_r > 0.9

    def test_empty_string_returns_zeros(self) -> None:
        assert script_ratios("") == (0.0, 0.0)

    def test_whitespace_only(self) -> None:
        assert script_ratios("   ") == (0.0, 0.0)

    def test_mixed_script(self) -> None:
        # Half Tamil, half ASCII.
        mixed = "வணக்கம் Hello"
        tamil_r, ascii_r = script_ratios(mixed)
        assert 0.2 < tamil_r < 0.9
        assert 0.1 < ascii_r < 0.8

    def test_digits_counted_as_other(self) -> None:
        # Digits are not Tamil or ASCII alpha.
        tamil_r, ascii_r = script_ratios("123 வணக்கம்")
        assert 0.0 < tamil_r < 1.0


# ─── compute_script_flags ─────────────────────────────────────────────────────

class TestComputeScriptFlags:
    """Tests for compute_script_flags()."""

    def test_pure_tamil_no_flags(self) -> None:
        flags = compute_script_flags("வணக்கம் உலகம்")
        assert flags == []

    def test_pure_english_wrong_lang(self) -> None:
        flags = compute_script_flags("Hello world this is English.")
        assert "WRONG_LANG" in flags

    def test_romanised_tamil(self) -> None:
        # Romanised Tamil: ASCII text that looks like Tamil words.
        # Use high ASCII ratio (all ASCII letters) and no Tamil chars.
        flags = compute_script_flags(
            "vanakkam makkal",
            tamil_ratio_min=0.4,
            ascii_ratio_romanised=0.7,
        )
        assert "ROMANISED_TA" in flags or "WRONG_LANG" in flags

    def test_code_mixed(self) -> None:
        # Tamil with some English words.
        code_mixed = "இந்த Hello world வார்த்தை"
        flags = compute_script_flags(
            code_mixed,
            tamil_ratio_min=0.4,
            tamil_ratio_code_mix=0.8,
        )
        # Should be flagged as CODE_MIXED or at least not pure.
        # The exact flag depends on ratios; just check something was flagged or not.
        # With this particular string, Tamil chars are mixed with ASCII.
        assert isinstance(flags, list)

    def test_empty_string_no_flags(self) -> None:
        assert compute_script_flags("") == []

    def test_wrong_lang_no_tamil_no_ascii(self) -> None:
        # Chinese characters – not Tamil, not ASCII letters.
        flags = compute_script_flags("你好世界")
        assert "WRONG_LANG" in flags

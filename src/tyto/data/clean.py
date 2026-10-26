"""Text cleaning for English and Tamil subtitle sentences.

Implements the TEXT RULES from PROJECT_BRIEF.md:

* Normalise Unicode to NFC.
* Strip subtitle markup tags (``<i>``, ``<b>``, ``<font …>`` etc.).
* Strip speaker dashes (lines starting with ``-`` or ``–``).
* Strip music symbols (``♪``, ``♫``, ``#`` when used as music note) and sound
  descriptions like ``[laughs]``, ``[MUSIC]``.
* Strip zero-width joiners and other invisible characters in Tamil.
* Keep the raw column untouched (caller is responsible for storing raw before
  calling this function).
* Flag wrong-language, code-mixed, and romanised-Tamil lines (do NOT delete).

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ─── Compile-once regexes ─────────────────────────────────────────────────────

# HTML/subtitle markup tags, e.g. <i>, </b>, <font color="red">, </font>
_TAG_RE = re.compile(r"<[^>]+>", re.IGNORECASE)

# Speaker label: leading dash (or em-dash) optionally with whitespace.
#   "- Hello" / "– Hello" / "-- Hello" / "-Hello"
_SPEAKER_DASH_RE = re.compile(r"^\s*[-\u2013\u2014]{1,2}\s*")

# Music and sound-effect markers.
#   ♪, ♫, lines enclosed in ♪…♪ or #…#
_MUSIC_SYMBOL_RE = re.compile(r"[♪♫]")
# Sound descriptions in brackets: [laughs], [MUSIC], (applause), etc.
_SOUND_DESC_RE = re.compile(r"[\[\(][A-Za-z\s]{1,40}[\]\)]")
# Hash signs used as music notes (standalone or paired), e.g. "# La la la #"
_HASH_MUSIC_RE = re.compile(r"\s*#\s*")

# Zero-width characters and invisible Unicode junk.
_ZW_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f"  # ZWS, ZWNJ, ZWJ, LRM, RLM
    r"\ufeff"                           # BOM
    r"\u00ad"                           # soft hyphen
    r"\u2060"                           # word joiner
    r"]"
)

# Double (or more) dashes: "--" not preceded/followed by a word character.
_DOUBLE_DASH_RE = re.compile(r"(?<!\w)--+(?!\w)")

# Collapse multiple whitespace (after all removals).
_SPACE_RE = re.compile(r"\s+")

# ─── Tamil Unicode block detection ────────────────────────────────────────────
# Tamil Unicode block: U+0B80–U+0BFF
_TAMIL_RANGE = range(0x0B80, 0x0C00)


def _char_is_tamil(ch: str) -> bool:
    return ord(ch) in _TAMIL_RANGE


def _char_is_ascii_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


def script_ratios(text: str) -> tuple[float, float]:
    """Return (tamil_ratio, ascii_ratio) for the alphabetic characters in *text*.

    Args:
        text: Input string (may be empty).

    Returns:
        Tuple of (tamil_char_ratio, ascii_letter_ratio).  Both are in [0, 1].
        Returns (0.0, 0.0) for empty or whitespace-only input.
    """
    alpha_chars = [ch for ch in text if not ch.isspace()]
    if not alpha_chars:
        return 0.0, 0.0
    n = len(alpha_chars)
    tamil_n = sum(1 for ch in alpha_chars if _char_is_tamil(ch))
    ascii_n = sum(1 for ch in alpha_chars if _char_is_ascii_letter(ch))
    return tamil_n / n, ascii_n / n


# ─── Public clean function ────────────────────────────────────────────────────

def clean_text(text: str, *, lang: str = "en") -> str:
    """Apply all text-cleaning rules to a raw subtitle sentence.

    The function is pure and deterministic: same input → same output.

    Steps applied (in order):

    1. NFC Unicode normalisation.
    2. Strip zero-width joiners / invisible chars (important for Tamil).
    3. Remove HTML/subtitle markup tags.
    4. Remove sound descriptions ``[laughs]`` / ``(applause)``.
    5. Remove music symbols ``♪`` / ``♫``.
    6. Remove leading speaker dashes ``-`` / ``–``.
    7. Replace ``--`` (double dash mid-line) with a single space.
    8. Collapse multiple whitespace and strip edges.

    Args:
        text: Raw subtitle sentence string.
        lang: Language hint (``"en"`` or ``"ta"``).  Reserved for
              future language-specific normalisation; currently unused.

    Returns:
        Cleaned sentence string.  May be empty if the entire sentence
        was markup or sound-effect annotations.
    """
    if not text:
        return ""

    # 1. NFC
    t = unicodedata.normalize("NFC", text)

    # 2. Strip zero-width / invisible characters (especially important for Tamil)
    t = _ZW_RE.sub("", t)

    # 3. Strip markup tags
    t = _TAG_RE.sub(" ", t)

    # 4. Strip sound descriptions
    t = _SOUND_DESC_RE.sub(" ", t)

    # 5. Strip music symbols
    t = _MUSIC_SYMBOL_RE.sub(" ", t)

    # 6. Strip leading speaker dashes (per line, in case of multi-line joined text)
    lines = t.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        line = _SPEAKER_DASH_RE.sub("", line)
        cleaned_lines.append(line)
    t = " ".join(cleaned_lines)

    # 7. Replace double dashes
    t = _DOUBLE_DASH_RE.sub(" ", t)

    # 8. Collapse whitespace
    t = _SPACE_RE.sub(" ", t).strip()

    return t


# ─── QC flag computation ──────────────────────────────────────────────────────

def compute_script_flags(
    ta_text: str,
    *,
    tamil_ratio_min: float = 0.4,
    tamil_ratio_code_mix: float = 0.8,
    ascii_ratio_romanised: float = 0.7,
) -> list[str]:
    """Return a list of script-based QC flag strings for a Tamil sentence.

    Flags produced:
    - ``WRONG_LANG``    – no Tamil characters at all.
    - ``ROMANISED_TA``  – ASCII ratio high, Tamil ratio very low.
    - ``CODE_MIXED``    – some Tamil chars but below code_mix threshold.

    Args:
        ta_text:               Cleaned Tamil sentence.
        tamil_ratio_min:       Below this → wrong language or romanised.
        tamil_ratio_code_mix:  Between min and this → code-mixed.
        ascii_ratio_romanised: ASCII ratio to classify as romanised Tamil.

    Returns:
        List of flag strings (may be empty).
    """
    flags: list[str] = []
    if not ta_text.strip():
        return flags

    tamil_r, ascii_r = script_ratios(ta_text)

    if tamil_r == 0.0:
        flags.append("WRONG_LANG")
    elif tamil_r < tamil_ratio_min:
        if ascii_r >= ascii_ratio_romanised:
            flags.append("ROMANISED_TA")
        else:
            flags.append("WRONG_LANG")
    elif tamil_r < tamil_ratio_code_mix:
        flags.append("CODE_MIXED")

    return flags

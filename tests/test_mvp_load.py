"""Unit tests for Project Tyto MVP Stage 2 v2018 loader."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tyto.mvp import load as load_mod


def _build_v2018_zip(
    zip_path: Path,
    records: list[tuple[str, str, str, str, str, str]],
) -> Path:
    """Create a synthetic v2018 Moses-format zip archive.

    Each record is ``(en_doc, ta_doc, en_sids, ta_sids, en_line, ta_line)``.
    """
    en_lines = [r[4] for r in records]
    ta_lines = [r[5] for r in records]
    ids_lines = [f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}" for r in records]

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("OpenSubtitles.en-ta.en", "\n".join(en_lines) + "\n")
        zf.writestr("OpenSubtitles.en-ta.ta", "\n".join(ta_lines) + "\n")
        zf.writestr("OpenSubtitles.en-ta.ids", "\n".join(ids_lines) + "\n")
    return zip_path


def _sample_valid_records() -> list[tuple[str, str, str, str, str, str]]:
    """Return 6 synthetic pairs across 3 distinct documents including a 1:2 link."""
    return [
        (
            "en/2015/111111/1.xml.gz",
            "ta/2015/111111/1.xml.gz",
            "1",
            "1",
            "Hello my friend, how are you?",
            "வணக்கம் நண்பா எப்படி இருக்கிறீர்கள்?",
        ),
        (
            "en/2015/111111/1.xml.gz",
            "ta/2015/111111/1.xml.gz",
            "2",
            "2",
            "I am fine, thank you.",
            "நான் நலமாக இருக்கிறேன் நன்றி.",
        ),
        (
            "en/2016/222222/2.xml.gz",
            "ta/2016/222222/2.xml.gz",
            "3",
            "3 4",
            "Wait for me outside the house.",
            "வீட்டிற்கு வெளியே எனக்காக காத்திருங்கள்.",
        ),
        (
            "en/2016/222222/2.xml.gz",
            "ta/2016/222222/2.xml.gz",
            "4",
            "5",
            "We need to leave immediately.",
            "நாம் உடனடியாக கிளம்ப வேண்டும்.",
        ),
        (
            "en/2017/6261062/3.xml.gz",
            "ta/2017/6261062/3.xml.gz",
            "1",
            "1",
            "Did you hear that sound?",
            "அந்த சத்தத்தை நீங்கள் கேட்டீர்களா?",
        ),
        (
            "en/2017/6261062/3.xml.gz",
            "ta/2017/6261062/3.xml.gz",
            "2",
            "2",
            "Everything will be alright.",
            "எல்லாம் சரியாகிவிடும்.",
        ),
    ]


def test_load_basic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load a valid mini corpus and check schema, row count, and metadata."""
    records = _sample_valid_records()
    zip_path = _build_v2018_zip(tmp_path / "en-ta.txt.zip", records)
    out_path = tmp_path / "pairs_v2018.parquet"

    monkeypatch.setattr(load_mod, "_EXPECTED_ROWS", len(records))
    df = load_mod.load_v2018(zip_path, out_path)

    expected_cols = {
        "pair_id",
        "film_id",
        "series_group",
        "year",
        "en_doc",
        "ta_doc",
        "en_sids",
        "ta_sids",
        "link_type",
        "en",
        "ta",
        "en_raw",
        "ta_raw",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == len(records)
    assert df["en_doc"].nunique() == 3
    assert out_path.exists()
    assert df.loc[df["film_id"] == "6261062", "series_group"].iloc[0] == "series_A"
    assert df.loc[df["film_id"] == "111111", "series_group"].iloc[0] == ""


def test_load_nfc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify NFC normalization and whitespace stripping while preserving raw text."""
    records = _sample_valid_records()
    # Inject denormalized NFD English ("e" + combining acute) and NFD Tamil (ெ + ா -> ொ)
    nfd_en = "  cafe\u0301 time  "
    nfd_ta = "  க\u0BC6\u0BBEடுங்கள் வணக்கம் நண்பா  "
    records[0] = (
        records[0][0],
        records[0][1],
        records[0][2],
        records[0][3],
        nfd_en,
        nfd_ta,
    )
    zip_path = _build_v2018_zip(tmp_path / "en-ta-nfc.zip", records)
    out_path = tmp_path / "pairs_nfc.parquet"

    monkeypatch.setattr(load_mod, "_EXPECTED_ROWS", len(records))
    df = load_mod.load_v2018(zip_path, out_path)

    assert df.loc[0, "en"] == "caf\u00e9 time"
    assert df.loc[0, "ta"] == "கொடுங்கள் வணக்கம் நண்பா"
    assert df.loc[0, "en_raw"] == nfd_en
    assert df.loc[0, "ta_raw"] == nfd_ta


def test_load_integrity_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A document with Tamil vowel-sign share < 0.10 must raise RuntimeError."""
    records = _sample_valid_records()
    # Replace doc 3 Tamil lines with pure ASCII (vowel_share = 0.0 < 0.10)
    records[4] = (
        records[4][0],
        records[4][1],
        records[4][2],
        records[4][3],
        "Did you hear that sound?",
        "corrupted ascii text instead of tamil",
    )
    records[5] = (
        records[5][0],
        records[5][1],
        records[5][2],
        records[5][3],
        "Everything will be alright.",
        "another corrupted ascii line",
    )
    zip_path = _build_v2018_zip(tmp_path / "en-ta-bad.zip", records)
    out_path = tmp_path / "pairs_bad.parquet"

    monkeypatch.setattr(load_mod, "_EXPECTED_ROWS", len(records))
    with pytest.raises(RuntimeError, match="vowel-sign gate"):
        load_mod.load_v2018(zip_path, out_path)


def test_load_link_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify link_type is '1:2' for a 1-en-to-2-ta alignment."""
    records = _sample_valid_records()
    zip_path = _build_v2018_zip(tmp_path / "en-ta-link.zip", records)
    out_path = tmp_path / "pairs_link.parquet"

    monkeypatch.setattr(load_mod, "_EXPECTED_ROWS", len(records))
    df = load_mod.load_v2018(zip_path, out_path)

    assert df.loc[2, "en_sids"] == "3"
    assert df.loc[2, "ta_sids"] == "3 4"
    assert df.loc[2, "link_type"] == "1:2"
    assert df.loc[0, "link_type"] == "1:1"


def test_load_row_count_gate(tmp_path: Path) -> None:
    """A corpus whose row count differs from expected 32,417 must raise RuntimeError."""
    records = _sample_valid_records()
    zip_path = _build_v2018_zip(tmp_path / "en-ta-short.zip", records)
    out_path = tmp_path / "pairs_short.parquet"

    with pytest.raises(RuntimeError, match="Row count mismatch"):
        load_mod.load_v2018(zip_path, out_path)

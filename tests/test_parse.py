"""Unit tests for tyto.data.parse.

Uses a minimal in-memory OPUS-style XML zip fixture so no network access is needed.
Verifies: schema compliance, timing extraction, metadata extraction, align_type.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tyto.data.parse import PAIRS_SCHEMA


def _gz(content: str) -> bytes:
    """Gzip-compress a string and return bytes."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(content.encode("utf-8"))
    return buf.getvalue()


def _make_mini_opus_zip(tmp_path: Path) -> Path:
    """Build a minimal OPUS XML zip with 2 en and 2 ta sentence files + alignment."""
    en_xml = """<?xml version="1.0" encoding="utf-8"?>
<corpus>
  <s id="1">
    <time id="T1S" value="00:00:01,000" type="S"/>
    <w id="1.1">Hello</w>
    <w id="1.2">world</w>
    <time id="T1E" value="00:00:03,000" type="E"/>
  </s>
  <s id="2">
    <time id="T2S" value="00:00:04,000" type="S"/>
    <w id="2.1">Goodbye</w>
    <time id="T2E" value="00:00:05,500" type="E"/>
  </s>
</corpus>
"""

    ta_xml = """<?xml version="1.0" encoding="utf-8"?>
<corpus>
  <s id="1">
    <w id="1.1">வணக்கம்</w>
    <w id="1.2">உலகம்</w>
  </s>
  <s id="2">
    <w id="2.1">விடை</w>
  </s>
</corpus>
"""

    align_xml = """<?xml version="1.0" encoding="utf-8"?>
<cesAlign version="1.0">
  <linkGrp targType="s" fromDoc="en/OpenSubtitles.en.ta.123.xml.gz"
           toDoc="ta/OpenSubtitles.en.ta.123.xml.gz">
    <link id="SL1" xtargets="1;1" type="1-1"/>
    <link id="SL2" xtargets="2;2" type="1-1"/>
  </linkGrp>
</cesAlign>
"""
    zip_path = tmp_path / "en-ta.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "OpenSubtitles/2016/xml/en/OpenSubtitles.en.ta.123.xml.gz",
            _gz(en_xml),
        )
        zf.writestr(
            "OpenSubtitles/2016/xml/ta/OpenSubtitles.en.ta.123.xml.gz",
            _gz(ta_xml),
        )
        zf.writestr(
            "OpenSubtitles/2016/xml/en-ta.xml.gz",
            _gz(align_xml),
        )
    return zip_path


class TestParseOpusZip:
    @pytest.fixture()
    def mini_zip(self, tmp_path: Path) -> Path:
        return _make_mini_opus_zip(tmp_path)

    def test_returns_dataframe(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert len(df) == 2

    def test_schema_columns_present(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        for col in PAIRS_SCHEMA.names:
            assert col in df.columns, f"Missing column: {col}"

    def test_english_text_extracted(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert "Hello" in df.iloc[0]["en_raw"]

    def test_tamil_text_extracted(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert "வணக்கம்" in df.iloc[0]["ta_raw"]

    def test_timing_extracted(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert df.iloc[0]["time_start"] == pytest.approx(1.0, abs=0.01)
        assert df.iloc[0]["time_end"] == pytest.approx(3.0, abs=0.01)

    def test_align_type_column(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert (df["align_type"] == "1-1").all()

    def test_source_version_stored(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert (df["source_version"] == "2016").all()

    def test_film_id_extracted(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert (df["film_id"] == "123").all()

    def test_max_pairs_limit(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016", max_pairs=1)
        assert len(df) == 1

    def test_pair_id_format(self, mini_zip: Path) -> None:
        from tyto.data.parse import parse_opus_zip

        df = parse_opus_zip(mini_zip, version="2016")
        assert df["pair_id"].str.contains(":").all()

    def test_parse_and_save(self, mini_zip: Path, tmp_path: Path) -> None:
        from tyto.data.parse import parse_and_save
        from tyto.utils.config import TytoConfig

        cfg = TytoConfig()
        out = tmp_path / "pairs.parquet"
        result = parse_and_save(cfg, mini_zip, output_path=out)
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 2

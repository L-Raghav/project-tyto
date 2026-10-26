"""CLI-level tests for the v2024 data commands.

These tests use Typer's CliRunner with:
- mocked HTTP (unittest.mock.patch) for download/list calls
- a small synthetic v2024-layout fixture on disk for parse

Each test fails explicitly if an option is missing or a path is stale.
No real network access; no real OPUS data.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from tyto.cli import app

runner = CliRunner()


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _gz(content: str) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(content.encode("utf-8"))
    return buf.getvalue()


def _make_v2024_fixture(root: Path) -> Path:
    """Write a minimal v2024 cache layout under *root*.

    Layout:
        <root>/v2024/
            en-ta.xml.gz         – alignment file
            en/
                OpenSubtitles.en.ta.99.xml.gz
            ta/
                OpenSubtitles.en.ta.99.xml.gz

    Returns the v2024 cache root (the directory passed to 'data parse --source').
    """
    cache = root / "v2024"
    en_dir = cache / "en"
    ta_dir = cache / "ta"
    en_dir.mkdir(parents=True)
    ta_dir.mkdir(parents=True)

    en_xml = """\
<?xml version="1.0" encoding="utf-8"?>
<corpus>
  <s id="1">
    <time id="T1S" value="00:00:01,000" type="S"/>
    <w id="1.1">Hello</w>
    <w id="1.2">world</w>
    <time id="T1E" value="00:00:03,000" type="E"/>
  </s>
  <s id="2">
    <w id="2.1">Goodbye</w>
  </s>
</corpus>
"""
    ta_xml = """\
<?xml version="1.0" encoding="utf-8"?>
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
    align_xml = """\
<?xml version="1.0" encoding="utf-8"?>
<cesAlign version="1.0">
  <linkGrp fromDoc="en/OpenSubtitles.en.ta.99.xml.gz"
           toDoc="ta/OpenSubtitles.en.ta.99.xml.gz">
    <link id="SL1" xtargets="1;1" type="1:1"/>
    <link id="SL2" xtargets="2;2" type="1:1"/>
  </linkGrp>
</cesAlign>
"""
    fname = "OpenSubtitles.en.ta.99.xml.gz"
    (en_dir / fname).write_bytes(_gz(en_xml))
    (ta_dir / fname).write_bytes(_gz(ta_xml))
    (cache / "en-ta.xml.gz").write_bytes(_gz(align_xml))

    # ta.zip is not needed for parse (files are already extracted),
    # but write it so download smoke-tests can reference it.
    ta_zip_path = cache / "ta.zip"
    with zipfile.ZipFile(ta_zip_path, "w") as zf:
        zf.writestr(fname, _gz(ta_xml))

    return cache


# ─── data --help ──────────────────────────────────────────────────────────────


def test_data_help_lists_all_commands() -> None:
    """data --help must list download, parse, qc, srt-align, audit, and list."""
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("download", "parse", "qc", "srt-align", "audit", "list"):
        assert cmd in result.output, (
            f"'data --help' is missing command '{cmd}'. Output:\n{result.output}"
        )


# ─── data download --help ─────────────────────────────────────────────────────


def test_download_help_shows_max_docs() -> None:
    """data download --help must expose --max-docs."""
    result = runner.invoke(app, ["data", "download", "--help"])
    assert result.exit_code == 0, result.output
    assert "--max-docs" in result.output, (
        f"--max-docs missing from download --help. Output:\n{result.output}"
    )


def test_download_help_version_text_correct() -> None:
    """--version help must mention 'latest' or 'v2024', not only 'e.g. 2016'."""
    result = runner.invoke(app, ["data", "download", "--help"])
    assert result.exit_code == 0, result.output
    # Must not still say the old "e.g. 2016" text.
    assert "e.g. 2016" not in result.output, (
        "Old '--version e.g. 2016' help text still present; should reference v2024/latest.\n"
        + result.output
    )
    # Must mention the new style.
    assert "v2024" in result.output or "latest" in result.output, (
        "Neither 'v2024' nor 'latest' appears in --version help text.\n" + result.output
    )


# ─── data parse --help ────────────────────────────────────────────────────────


def test_parse_help_shows_source_and_max_docs() -> None:
    """data parse --help must expose --source and --max-docs (v2024 options)."""
    result = runner.invoke(app, ["data", "parse", "--help"])
    assert result.exit_code == 0, result.output
    assert "--source" in result.output, (
        f"--source missing from parse --help. Output:\n{result.output}"
    )
    assert "--max-docs" in result.output, (
        f"--max-docs missing from parse --help. Output:\n{result.output}"
    )


def test_parse_help_no_stale_zip_reference() -> None:
    """--zip must not appear in parse --help (removed stale single-zip path)."""
    result = runner.invoke(app, ["data", "parse", "--help"])
    assert result.exit_code == 0, result.output
    assert "--zip" not in result.output, (
        "Stale '--zip' option still present in 'data parse --help'.\n" + result.output
    )


# ─── data list ────────────────────────────────────────────────────────────────


def test_list_help_accessible() -> None:
    """data list --help must succeed."""
    result = runner.invoke(app, ["data", "list", "--help"])
    assert result.exit_code == 0, result.output
    assert "--all" in result.output


def test_list_with_mocked_api(tmp_path: Path) -> None:
    """data list returns a table when the OPUS API returns valid JSON."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "corpora": [
            {
                "version": "v2024",
                "preprocessing": "xml",
                "size": 17_000_000,
                "url": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/xml/en-ta.xml.gz",
            }
        ]
    }
    fake_response.raise_for_status = MagicMock()

    # requests is imported inside list_corpus_files; patch the real module.
    with patch("requests.get", return_value=fake_response):
        result = runner.invoke(app, ["data", "list"])
    assert result.exit_code == 0, result.output
    assert "v2024" in result.output


def test_list_network_failure_is_graceful() -> None:
    """data list must not crash when the OPUS API is unreachable."""
    import requests as req_mod

    # requests is imported inside the function body; patch the real module.
    with patch("requests.get", side_effect=req_mod.ConnectionError("timeout")):
        result = runner.invoke(app, ["data", "list"])
    assert result.exit_code == 0, result.output
    # Should print 'No releases found' gracefully.
    assert "No releases found" in result.output or "network" in result.output.lower()


# ─── data download (mocked HTTP) ─────────────────────────────────────────────


def test_download_max_docs_end_to_end(tmp_path: Path) -> None:
    """data download --max-docs 2 runs the full flow with mocked HTTP.

    Strategy: pre-write the alignment .xml.gz into the cache dir so that
    _verify_size sees a size match and collect_en_doc_paths can parse it.
    Only the Range probe, ta.zip download, and RemoteZip are mocked.
    """
    import yaml

    ta_fname = "OpenSubtitles.en.ta.1.xml.gz"
    align_xml = f"""\
<?xml version="1.0" encoding="utf-8"?>
<cesAlign version="1.0">
  <linkGrp fromDoc="en/{ta_fname}" toDoc="ta/{ta_fname}">
    <link id="SL1" xtargets="1;1" type="1:1"/>
  </linkGrp>
</cesAlign>
"""
    align_gz_bytes = _gz(align_xml)

    # Pre-seed the cache dir.
    version = "v2024"
    cache_root = tmp_path / "data" / "raw" / "opus" / version
    cache_root.mkdir(parents=True)
    align_local = cache_root / "en-ta.xml.gz"
    align_local.write_bytes(align_gz_bytes)

    # Build a ta.zip in memory.
    ta_zip_buf = io.BytesIO()
    with zipfile.ZipFile(ta_zip_buf, "w") as zf:
        zf.writestr(ta_fname, _gz("<corpus/>"))
    ta_zip_bytes = ta_zip_buf.getvalue()

    def fake_get(url, **kwargs):
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.raise_for_status = MagicMock()
        headers = kwargs.get("headers", {})
        if "Range" in headers and headers["Range"] == "bytes=0-0":
            resp.status_code = 206
            return resp
        resp.status_code = 200
        resp.headers = {"content-length": str(len(ta_zip_bytes))}
        resp.iter_content = lambda chunk_size=None: iter([ta_zip_bytes])
        return resp

    def fake_head(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.status_code = 200
        # Match exact alignment file size so _verify_size skips re-download.
        resp.headers = {"content-length": str(len(align_gz_bytes))}
        return resp

    remote_zip_ctx = MagicMock()
    remote_zip_ctx.__enter__ = lambda s: s
    remote_zip_ctx.__exit__ = MagicMock(return_value=False)
    remote_zip_ctx.namelist.return_value = [f"en/{ta_fname}"]
    remote_zip_ctx.read.return_value = _gz("<corpus/>")

    from tyto.data.download import extract_tgt_docs as real_extract_tgt_docs

    def check_extract(tgt_zip_path, *args, **kwargs):
        assert tgt_zip_path.exists()
        assert tgt_zip_path.stat().st_size == len(ta_zip_bytes)
        return real_extract_tgt_docs(tgt_zip_path, *args, **kwargs)

    # Point download_dir at our tmp tree via a minimal config override.
    opus_dir = str(tmp_path / "data" / "raw" / "opus").replace("\\", "/")
    cfg_path = tmp_path / "test_dl.yaml"
    cfg_path.write_text(
        yaml.dump({"opus": {"download_dir": opus_dir}}),
        encoding="utf-8",
    )

    with (
        patch("requests.get", side_effect=fake_get),
        patch("requests.head", side_effect=fake_head),
        patch("remotezip.RemoteZip", return_value=remote_zip_ctx),
        patch("tyto.data.download.extract_tgt_docs", side_effect=check_extract),
    ):
        result = runner.invoke(
            app,
            [
                "data", "download",
                "--config", str(cfg_path),
                "--max-docs", "2",
                "--version", version,
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "alignment" in result.output or "en_docs" in result.output
    ta_zip_dest = cache_root / "ta.zip"
    assert ta_zip_dest.exists()
    assert ta_zip_dest.stat().st_size == len(ta_zip_bytes)


# ─── data parse (v2024 fixture) ───────────────────────────────────────────────


def test_parse_v2024_fixture_end_to_end(tmp_path: Path) -> None:
    """data parse --source <fixture> writes a valid pairs.parquet."""
    import pandas as pd

    cache = _make_v2024_fixture(tmp_path)
    out = tmp_path / "pairs.parquet"

    result = runner.invoke(
        app,
        [
            "data", "parse",
            "--source", str(cache),
            "--output", str(out),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert out.exists(), "pairs.parquet was not created"

    df = pd.read_parquet(out)
    assert len(df) == 2, f"Expected 2 pairs, got {len(df)}"
    assert "99" in df["film_id"].iloc[0]
    assert "Hello" in df["en_raw"].iloc[0]
    assert "வணக்கம்" in df["ta_raw"].iloc[0]


def test_parse_v2024_max_docs(tmp_path: Path) -> None:
    """--max-docs 1 on a fixture with 1 document returns pairs from that doc."""
    import pandas as pd

    cache = _make_v2024_fixture(tmp_path)
    out = tmp_path / "pairs.parquet"

    result = runner.invoke(
        app,
        [
            "data", "parse",
            "--source", str(cache),
            "--output", str(out),
            "--max-docs", "1",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    df = pd.read_parquet(out)
    assert len(df) > 0


def test_parse_missing_source_exits_nonzero(tmp_path: Path) -> None:
    """data parse --source <nonexistent> must exit 1."""
    result = runner.invoke(
        app,
        ["data", "parse", "--source", str(tmp_path / "does_not_exist")],
    )
    assert result.exit_code == 1


def test_parse_missing_cache_dir_exits_nonzero() -> None:
    """data parse with no --source and a missing cache dir must exit 1."""
    # Config points to download_dir=data/raw/opus; v2024 subdir won't exist in CI.
    # Temporarily patch the cache check to guarantee it fails.
    with patch("tyto.cli.Path.exists", return_value=False):
        result = runner.invoke(app, ["data", "parse"])
    assert result.exit_code == 1

"""CLI tests for Project Tyto MVP sub-commands."""

from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from tyto.cli import app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes from CLI output."""
    return _ANSI_RE.sub("", text)


def test_mvp_help() -> None:
    """`python -m tyto mvp --help` lists all 6 MVP sub-commands."""
    result = runner.invoke(app, ["mvp", "--help"])
    assert result.exit_code == 0
    out = _plain(result.output)
    for cmd in ("load", "features", "label", "mine", "predict", "export"):
        assert cmd in out


def test_mvp_load_missing_file(tmp_path: Path) -> None:
    """`mvp load` with a nonexistent zip exits with non-zero code."""
    missing = tmp_path / "nonexistent.zip"
    result = runner.invoke(app, ["mvp", "load", "--zip", str(missing)])
    assert result.exit_code != 0


def test_mvp_features_missing_pairs(tmp_path: Path) -> None:
    """`mvp features` with no pairs_v2018.parquet fails with an error."""
    missing_pairs = tmp_path / "pairs_v2018.parquet"
    result = runner.invoke(app, ["mvp", "features", "--pairs", str(missing_pairs)])
    assert result.exit_code != 0
    assert result.exception is not None or "error" in _plain(result.output).lower()

"""Unit tests for Typer CLI commands and help output."""

from typer.testing import CliRunner

from tyto.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Ensure root CLI help lists all required groups and commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = result.output

    assert "doctor" in output
    assert "data" in output
    assert "features" in output
    assert "mine" in output
    assert "train" in output
    assert "app" in output
    assert "export" in output


def test_cli_unimplemented_groups() -> None:
    """Ensure placeholder command groups print 'not implemented yet'."""
    for group_name in ["data", "features", "mine", "train", "app", "export"]:
        result = runner.invoke(app, [group_name])
        assert result.exit_code == 0
        assert "not implemented yet" in result.output


def test_cli_doctor_help() -> None:
    """Ensure doctor command help lists expected flags."""
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--cpu-ok" in result.output


def test_cli_doctor_fake_smoke() -> None:
    """Run doctor using the fake model and cpu-ok to ensure clean execution."""
    result = runner.invoke(app, ["doctor", "--cpu-ok", "--smoke-model", "fake"])
    assert result.exit_code == 0
    assert "Doctor check completed successfully" in result.output

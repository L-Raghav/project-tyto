"""Shared pytest configuration and fixtures for Project Tyto tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _deterministic_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable ANSI colors and fix terminal width for deterministic CLI output."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


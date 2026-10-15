"""Unit tests for configuration loader and Pydantic validation."""

from pathlib import Path

import pytest

from tyto.utils.config import TytoConfig, load_config


def test_load_default_config() -> None:
    """Ensure the default configuration file loads and validates cleanly."""
    cfg = load_config("configs/default.yaml")

    assert isinstance(cfg, TytoConfig)
    assert cfg.seed == 42
    assert cfg.device.use_cuda is True
    assert cfg.paths.data_dir == Path("data")
    assert cfg.paths.resources_dir == Path("data/resources")
    assert cfg.paths.gold_dir == Path("data/gold")
    assert "IDIOM_LITERAL" in cfg.taxonomy.labels
    assert "LOCALISED_OK" in cfg.taxonomy.labels
    assert cfg.taxonomy.severities == [0, 1, 2]
    assert cfg.models.labse.name == "sentence-transformers/LaBSE"


def test_load_config_missing_file() -> None:
    """Ensure FileNotFoundError is raised when configuration file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_config("configs/non_existent_path.yaml")


def test_load_custom_config(tmp_path: Path) -> None:
    """Ensure custom YAML configurations load and override defaults."""
    custom_yaml = tmp_path / "custom.yaml"
    custom_yaml.write_text(
        "seed: 123\ndevice:\n  use_cuda: false\n  batch_size: 16\n",
        encoding="utf-8",
    )

    cfg = load_config(custom_yaml)
    assert cfg.seed == 123
    assert cfg.device.use_cuda is False
    assert cfg.device.batch_size == 16
    assert cfg.paths.data_dir == Path("data")

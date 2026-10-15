"""Configuration loader and schema validation using Pydantic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PathsConfig(BaseModel):
    """Filesystem paths for datasets, artifacts, and models."""

    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    gold_dir: Path = Path("data/gold")
    resources_dir: Path = Path("data/resources")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


class DeviceConfig(BaseModel):
    """Device, precision, and batching configuration."""

    use_cuda: bool = True
    fp16: bool = True
    batch_size: int = 32
    backoff_factor: int = 2
    num_workers: int = 0


class ModelEntryConfig(BaseModel):
    """Configuration for an individual model."""

    name: str
    max_seq_length: int = 128
    device: str = "cuda"


class ModelsConfig(BaseModel):
    """Mapping of registered model configurations."""

    labse: ModelEntryConfig = Field(
        default_factory=lambda: ModelEntryConfig(
            name="sentence-transformers/LaBSE",
            max_seq_length=128,
            device="cuda",
        )
    )


class TaxonomyConfig(BaseModel):
    """Translation error taxonomy labels and severities."""

    labels: list[str] = Field(
        default_factory=lambda: [
            "OK",
            "LOCALISED_OK",
            "IDIOM_LITERAL",
            "TONE_SOFTENED",
            "TONE_INTENSIFIED",
            "OMISSION",
            "ADDITION",
            "MISTRANSLATION",
            "REGISTER_MISMATCH",
            "MISALIGNED",
        ]
    )
    severities: list[int] = Field(default_factory=lambda: [0, 1, 2])
    issue_labels: list[str] = Field(
        default_factory=lambda: [
            "IDIOM_LITERAL",
            "TONE_SOFTENED",
            "TONE_INTENSIFIED",
            "OMISSION",
            "ADDITION",
            "MISTRANSLATION",
            "REGISTER_MISMATCH",
        ]
    )


class TytoConfig(BaseModel):
    """Root configuration for Project Tyto."""

    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)


def load_config(path: str | Path = "configs/default.yaml") -> TytoConfig:
    """Load and validate Tyto configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated TytoConfig instance.

    Raises:
        FileNotFoundError: If the specified configuration file does not exist.
        ValueError: If YAML parsing or validation fails.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data: Any = yaml.safe_load(f) or {}

    logger.debug("Loaded configuration from %s", config_path)
    return TytoConfig.model_validate(data)

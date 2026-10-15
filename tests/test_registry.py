"""Unit tests for ModelRegistry with fake tiny model and memory cleanup."""

import numpy as np
import pytest
import torch

from tyto.modeling.registry import FakeTinyModel, ModelRegistry, default_registry


def test_registry_registration() -> None:
    """Check that default models (labse, fake) are properly registered."""
    registered = default_registry.list_registered()
    assert "labse" in registered
    assert "fake" in registered


def test_fake_model_lazy_load_and_unload() -> None:
    """Check lazy loading, inference, and unload behavior with FakeTinyModel."""
    registry = ModelRegistry()
    registry.register(
        "test_fake",
        lambda device="cpu", fp16=False, **kw: FakeTinyModel(embedding_dim=16).to(device),
        is_large=False,
    )

    assert not registry.is_loaded("test_fake")

    model = registry.get("test_fake", device="cpu")
    assert registry.is_loaded("test_fake")
    assert isinstance(model, FakeTinyModel)

    texts = ["First line", "Second line"]
    embs = model.encode(texts)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (2, 16)

    tensor_embs = model.encode(texts, convert_to_tensor=True)
    assert isinstance(tensor_embs, torch.Tensor)
    assert tensor_embs.shape == (2, 16)

    registry.unload("test_fake")
    assert not registry.is_loaded("test_fake")


def test_registry_unregistered_model() -> None:
    """Ensure accessing an unregistered model raises KeyError."""
    registry = ModelRegistry()
    with pytest.raises(KeyError, match="not registered"):
        registry.get("unknown_model")


def test_vram_report_format() -> None:
    """Ensure vram_report returns a valid structure."""
    report = default_registry.vram_report()
    assert isinstance(report, dict)
    assert "cuda_available" in report
    assert "total_mb" in report
    assert "free_mb" in report
    assert "allocated_mb" in report
    assert "reserved_mb" in report

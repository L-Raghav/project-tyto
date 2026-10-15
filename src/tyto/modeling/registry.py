"""Model registry with lazy loading, explicit memory release, and VRAM monitoring."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FakeTinyModel(torch.nn.Module):
    """A lightweight mock encoder for fast, deterministic unit tests."""

    def __init__(self, embedding_dim: int = 16) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = torch.nn.Linear(embedding_dim, embedding_dim)

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str | torch.device | None = None,
        normalize_embeddings: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Mock encoding method compatible with SentenceTransformer.encode."""
        if isinstance(sentences, str):
            single = True
            texts = [sentences]
        else:
            single = False
            texts = list(sentences)

        tensors = []
        for text in texts:
            val = float(len(text) % 100) / 100.0
            vec = torch.full((self.embedding_dim,), val, dtype=torch.float32)
            tensors.append(vec)

        batch = torch.stack(tensors)
        param = next(self.parameters())
        batch = batch.to(device=param.device, dtype=param.dtype)
        out = self.linear(batch)

        if normalize_embeddings:
            out = torch.nn.functional.normalize(out, p=2, dim=1)

        if convert_to_tensor:
            return out[0] if single else out

        out_np = out.detach().cpu().numpy()
        return out_np[0] if single else out_np


class ModelRegistry:
    """Registry managing lazy-loaded models, GPU memory residency, and clean unloads."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}
        self._loaded: dict[str, Any] = {}
        self._loaded_devices: dict[str, str] = {}
        self._is_large: dict[str, bool] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., Any],
        is_large: bool = True,
    ) -> None:
        """Register a model loader factory.

        Args:
            name: Identifier for the model.
            factory: Callable accepting (device=..., fp16=...) returning the model.
            is_large: Whether this model occupies significant GPU VRAM.
        """
        self._factories[name] = factory
        self._is_large[name] = is_large
        logger.debug("Registered model factory '%s' (is_large=%s)", name, is_large)

    def is_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded in memory."""
        return name in self._loaded

    def list_registered(self) -> list[str]:
        """Return sorted list of registered model identifiers."""
        return sorted(self._factories.keys())

    def list_loaded(self) -> list[str]:
        """Return sorted list of currently loaded model identifiers."""
        return sorted(self._loaded.keys())

    def get(
        self,
        name: str,
        device: str | torch.device | None = None,
        fp16: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Get or lazily load a registered model.

        If loading a large model on GPU and another large model is already loaded on GPU,
        the existing model is unloaded first to preserve 8 GB VRAM budget.

        Args:
            name: Identifier of the registered model.
            device: Target device ("cuda", "cpu", or None for auto).
            fp16: Whether to convert the model to half precision on GPU.
            kwargs: Extra parameters forwarded to the model factory.

        Returns:
            Loaded model instance.
        """
        if name not in self._factories:
            raise KeyError(
                f"Model '{name}' is not registered. Registered models: {self.list_registered()}"
            )

        target_device = (
            str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if name in self._loaded:
            current_device = self._loaded_devices.get(name, "cpu")
            if target_device.startswith("cuda") and current_device.startswith("cuda"):
                return self._loaded[name]
            if target_device == "cpu" and current_device == "cpu":
                return self._loaded[name]
            self.unload(name)

        if target_device.startswith("cuda") and self._is_large.get(name, True):
            for loaded_name in list(self._loaded.keys()):
                is_on_cuda = self._loaded_devices.get(loaded_name, "").startswith("cuda")
                is_large_model = self._is_large.get(loaded_name, True)
                if is_on_cuda and is_large_model:
                    logger.info(
                        "Unloading large model '%s' to make room for '%s' on GPU",
                        loaded_name,
                        name,
                    )
                    self.unload(loaded_name)

        factory = self._factories[name]
        logger.info("Lazily loading model '%s' on device '%s' (fp16=%s)", name, target_device, fp16)
        model = factory(device=target_device, fp16=fp16, **kwargs)

        self._loaded[name] = model
        self._loaded_devices[name] = target_device
        return model

    def unload(self, name: str | None = None) -> None:
        """Explicitly unload one or all models and free CUDA cache.

        Args:
            name: Model name to unload, or None to unload all loaded models.
        """
        if name is not None:
            if name in self._loaded:
                logger.info("Unloading model '%s'", name)
                del self._loaded[name]
                self._loaded_devices.pop(name, None)
        else:
            logger.info("Unloading all models (%s)", list(self._loaded.keys()))
            self._loaded.clear()
            self._loaded_devices.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def vram_report(self) -> dict[str, Any]:
        """Generate a VRAM usage report.

        Returns:
            Dictionary with CUDA availability, device name, and memory stats in MB.
        """
        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "gpu_name": None,
                "total_mb": 0.0,
                "free_mb": 0.0,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
            }

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated_bytes = torch.cuda.memory_allocated()
        reserved_bytes = torch.cuda.memory_reserved()

        return {
            "cuda_available": True,
            "device_index": 0,
            "gpu_name": torch.cuda.get_device_name(0),
            "total_mb": round(total_bytes / (1024**2), 2),
            "free_mb": round(free_bytes / (1024**2), 2),
            "allocated_mb": round(allocated_bytes / (1024**2), 2),
            "reserved_mb": round(reserved_bytes / (1024**2), 2),
        }


def _load_labse(
    device: str = "cuda",
    fp16: bool = True,
    **kwargs: Any,
) -> Any:
    """Load LaBSE SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformers/LaBSE on %s", device)
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device, **kwargs)
    if device.startswith("cuda") and fp16:
        model.half()
    return model


def _load_fake(
    device: str = "cpu",
    fp16: bool = False,
    embedding_dim: int = 16,
    **kwargs: Any,
) -> FakeTinyModel:
    """Load FakeTinyModel for testing."""
    model = FakeTinyModel(embedding_dim=embedding_dim).to(device)
    if device.startswith("cuda") and fp16:
        model.half()
    return model


default_registry = ModelRegistry()
default_registry.register("labse", _load_labse, is_large=True)
default_registry.register("fake", _load_fake, is_large=False)

"""Unit tests for seed reproducibility utility."""

import numpy as np
import torch

from tyto.utils.seed import set_seed


def test_set_seed_reproducibility() -> None:
    """Ensure set_seed produces deterministic pseudo-random sequences."""
    set_seed(42)
    val_np_1 = np.random.rand(5)
    val_torch_1 = torch.rand(5)

    set_seed(42)
    val_np_2 = np.random.rand(5)
    val_torch_2 = torch.rand(5)

    np.testing.assert_allclose(val_np_1, val_np_2)
    assert torch.equal(val_torch_1, val_torch_2)

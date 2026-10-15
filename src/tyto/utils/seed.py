"""Seed utilities for reproducible experiments across Python, NumPy, and PyTorch."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic_torch: bool = False) -> int:
    """Set random seed across Python random, NumPy, and PyTorch.

    Args:
        seed: Integer seed value to set.
        deterministic_torch: If True, forces cuDNN deterministic mode.

    Returns:
        The integer seed value applied.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("Random seed initialized to %d (deterministic=%s)", seed, deterministic_torch)
    return seed

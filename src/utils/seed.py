"""Global seed control for reproducibility."""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA + MPS).

    Also sets ``PYTHONHASHSEED`` and forces deterministic cuDNN.
    Optional backends (NumPy, PyTorch, CUDA, MPS) are only seeded when
    importable / available, so this also works in CPU-only environments.

    Args:
        seed: Non-negative integer seed.

    Raises:
        ValueError: If ``seed`` is negative.
    """

    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed!r}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.debug("numpy not installed; skipping")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            mps_module = getattr(torch, "mps", None)
            if mps_module is not None and hasattr(mps_module, "manual_seed"):
                mps_module.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not installed; skipping")

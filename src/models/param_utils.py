"""Parameter-counting utilities for model complexity analysis."""

from __future__ import annotations

import torch.nn as nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts for *model*.

    Works for both vanilla ``nn.Module`` and ``peft.PeftModel`` instances.
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

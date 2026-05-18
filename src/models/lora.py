"""LoRA adapter injection for DistilBERT QA."""

from __future__ import annotations

from typing import Any


def apply_lora(model: Any, rank: int) -> Any:
    """Attach LoRA adapters to a DistilBERT QA model.

    Args:
        model: Model returned by ``build_model``.
        rank: LoRA rank (see ``config.LORA_RANKS``).

    Returns:
        The model with LoRA adapters injected and only the adapter
        parameters (plus the QA head) marked trainable.
    """

    # TODO(stage 3): wrap with peft.get_peft_model using LoraConfig(r=rank).
    raise NotImplementedError

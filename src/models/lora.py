"""LoRA adapter injection for DistilBERT QA."""

from __future__ import annotations

import logging

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel

import config

logger = logging.getLogger(__name__)


def apply_lora(model: PreTrainedModel, rank: int) -> PeftModel:
    """Attach LoRA adapters to a DistilBERT QA model.

    The base model is fully frozen; only the injected LoRA parameters
    (and the QA classification head) are marked trainable.

    Args:
        model: Model returned by :func:`src.models.distilbert_qa.build_model`.
        rank: LoRA rank (see ``config.LORA_RANKS``).

    Returns:
        A :class:`peft.PeftModel` wrapping the original model with LoRA
        adapters on the attention projections specified by
        ``config.LORA_TARGET_MODULES``.

    Raises:
        ValueError: If any configured target module is absent from *model*.
    """

    _validate_target_modules(model)

    lora_alpha = config.LORA_ALPHA_MULTIPLIER * rank
    lora_config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
    )

    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.requires_grad for p in peft_model.parameters())
    assert trainable > 0, "No trainable parameters after applying LoRA"

    return peft_model


def _validate_target_modules(model: PreTrainedModel) -> None:
    """Raise if any target module name is not present in the model."""

    all_names = {name.split(".")[-1] for name, _ in model.named_modules()}
    missing = set(config.LORA_TARGET_MODULES) - all_names
    if missing:
        raise ValueError(
            f"LoRA target modules not found in model: {sorted(missing)}. "
            f"Available leaf module names: {sorted(all_names)}"
        )

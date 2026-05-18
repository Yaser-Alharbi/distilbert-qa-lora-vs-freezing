"""Training loop for DistilBERT QA fine-tuning."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def train(model: Any, data: Any, config: Any) -> Tuple[Any, Dict[str, Any]]:
    """Fine-tune ``model`` on ``data``.

    Args:
        model: A DistilBERT QA model (optionally with LoRA adapters).
        data: ``(train_dataset, validation_dataset)`` from ``load_squad``.
        config: Project ``config`` module (reads ``BATCH_SIZE``, ``LR``,
            ``EPOCHS``, ``WEIGHT_DECAY``, ``WARMUP_RATIO``, ``DEVICE``).

    Returns:
        ``(trained_model, history)``, where ``history`` has keys
        ``train_loss``, ``val_loss``, ``val_em``, ``val_f1`` mapping to
        per-epoch lists.
    """

    # TODO(stage 3): AdamW + warmup-linear schedule, per-epoch eval and logging.
    raise NotImplementedError

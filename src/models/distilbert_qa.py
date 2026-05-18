"""DistilBERT span-prediction model construction."""

from __future__ import annotations

from typing import Any


def build_model(freeze_config: str) -> Any:
    """Load DistilBERT for QA and apply a layer-freezing scheme.

    Args:
        freeze_config: Key into ``config.FREEZE_CONFIGS``
            (``C0`` head-only, ``C1`` top-1, ``C2`` top-3, ``C3`` full).

    Returns:
        A ``transformers.PreTrainedModel`` with ``requires_grad`` set
        according to ``freeze_config``.
    """

    # TODO(stage 3): load DistilBertForQuestionAnswering and freeze per config.
    raise NotImplementedError

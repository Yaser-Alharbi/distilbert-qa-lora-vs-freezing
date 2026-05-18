"""DistilBERT span-prediction model construction with layer-freezing strategies."""

from __future__ import annotations

import logging

from transformers import AutoModelForQuestionAnswering, PreTrainedModel

import config

logger = logging.getLogger(__name__)

_NUM_TRANSFORMER_LAYERS = 6

_STRATEGY_TO_TRAINABLE_LAYERS: dict[str, int] = {
    "head_only": 0,
    "top_1": 1,
    "top_3": 3,
    "full": _NUM_TRANSFORMER_LAYERS,
}


def build_model(freeze_config: str) -> PreTrainedModel:
    """Load DistilBERT for QA and apply a layer-freezing scheme.

    Args:
        freeze_config: Key into ``config.FREEZE_CONFIGS``
            (``C0`` head-only, ``C1`` top-1, ``C2`` top-3, ``C3`` full).

    Returns:
        A ``transformers.PreTrainedModel`` with ``requires_grad`` set
        according to ``freeze_config``.

    Raises:
        ValueError: If *freeze_config* is not a recognised key.
    """

    if freeze_config not in config.FREEZE_CONFIGS:
        raise ValueError(
            f"Unknown freeze_config {freeze_config!r}; "
            f"expected one of {list(config.FREEZE_CONFIGS)}"
        )

    strategy = config.FREEZE_CONFIGS[freeze_config]
    model = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_NAME)
    _apply_freezing(model, strategy)
    return model


def _apply_freezing(model: PreTrainedModel, strategy: str) -> None:
    """Set ``requires_grad`` on every parameter according to *strategy*."""

    if strategy not in _STRATEGY_TO_TRAINABLE_LAYERS:
        raise ValueError(
            f"Unknown strategy {strategy!r}; "
            f"expected one of {list(_STRATEGY_TO_TRAINABLE_LAYERS)}"
        )

    trainable_layer_count = _STRATEGY_TO_TRAINABLE_LAYERS[strategy]
    is_full = strategy == "full"

    for param in model.parameters():
        param.requires_grad = False

    for param in model.qa_outputs.parameters():
        param.requires_grad = True

    if is_full:
        for param in model.distilbert.parameters():
            param.requires_grad = True
    else:
        layers = model.distilbert.transformer.layer
        assert len(layers) == _NUM_TRANSFORMER_LAYERS, (
            f"Expected {_NUM_TRANSFORMER_LAYERS} transformer layers, got {len(layers)}"
        )
        for layer in layers[-trainable_layer_count:] if trainable_layer_count > 0 else []:
            for param in layer.parameters():
                param.requires_grad = True

    _assert_trainable_layer_count(model, strategy, trainable_layer_count)


def _assert_trainable_layer_count(
    model: PreTrainedModel,
    strategy: str,
    expected_trainable_layers: int,
) -> None:
    """Verify exactly the expected transformer layers are trainable."""

    layers = model.distilbert.transformer.layer
    actual = sum(
        1 for layer in layers if any(p.requires_grad for p in layer.parameters())
    )

    if strategy == "full":
        assert actual == _NUM_TRANSFORMER_LAYERS, (
            f"Strategy 'full': expected all {_NUM_TRANSFORMER_LAYERS} layers trainable, "
            f"got {actual}"
        )
    else:
        assert actual == expected_trainable_layers, (
            f"Strategy {strategy!r}: expected {expected_trainable_layers} trainable "
            f"transformer layers, got {actual}"
        )

    embeddings_trainable = any(
        p.requires_grad for p in model.distilbert.embeddings.parameters()
    )
    if strategy == "full":
        assert embeddings_trainable, "Strategy 'full' must leave embeddings trainable"
    else:
        assert not embeddings_trainable, (
            f"Strategy {strategy!r} must freeze embeddings"
        )

    assert all(p.requires_grad for p in model.qa_outputs.parameters()), (
        "QA head must always be trainable"
    )

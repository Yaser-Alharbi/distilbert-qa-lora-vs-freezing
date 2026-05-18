"""Report figures.

All functions save to ``config.PLOTS_DIR``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def save_learning_curve(history: Dict[str, List[float]], filename: str) -> Path:
    """Save train/validation loss and F1 curves.

    Args:
        history: Per-epoch metrics from ``trainer.train``.
        filename: Output filename (no directory component).

    Returns:
        Absolute path to the saved PNG.
    """

    # TODO(stage 5): plot loss and F1 vs. epoch, savefig, plt.close.
    raise NotImplementedError


def save_pareto(results: List[Dict[str, Any]], filename: str) -> Path:
    """Save a Pareto plot of F1 against trainable-parameter count.

    Args:
        results: Per-run dicts with at least ``f1``, ``trainable_params``,
            ``condition``.
        filename: Output filename (no directory component).

    Returns:
        Absolute path to the saved PNG.
    """

    # TODO(stage 5): scatter F1 vs. trainable_params, label points by condition.
    raise NotImplementedError


def save_tsne(embeddings: Any, labels: List[str], filename: str) -> Path:
    """Save a 2-D t-SNE projection of model embeddings.

    Args:
        embeddings: ``(N, D)`` array of embeddings.
        labels: One label per row.
        filename: Output filename (no directory component).

    Returns:
        Absolute path to the saved PNG.
    """

    # TODO(stage 5): fit t-SNE with fixed random_state, scatter by label.
    raise NotImplementedError

"""Thin dispatchers that keep the legacy ``save_*`` surface intact.

Stage 5 figures are implemented in dedicated modules; these wrappers
exist so any code that imports the old function names continues to work.
The new modules should be preferred for new code:

* :mod:`src.plotting.learning_curves`
* :mod:`src.plotting.pareto`
* :mod:`src.plotting.qtype_heatmap`
* :mod:`src.plotting.lowdim`
* :mod:`src.plotting.calibration`
* :mod:`src.plotting.error_analysis`
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import config
from src.plotting import learning_curves, lowdim, pareto

logger = logging.getLogger(__name__)


def save_learning_curve(history: Dict[str, List[float]], filename: str) -> Path:
    """Render the Stage 5 small-multiples grid + val overlay.

    Args:
        history: Ignored. Stage 5 reads per-run ``history.json`` files
            directly from :data:`config.RUNS_DIR`.
        filename: Ignored. Output stems are fixed inside the new module.

    Returns:
        Absolute path to the small-multiples grid figure (the overlay is
        written alongside it).
    """

    if history or filename:
        logger.debug("save_learning_curve: legacy args ignored; reading %s", config.RUNS_DIR)
    paths = learning_curves.save_learning_curves()
    return paths[0]


def save_pareto(results: List[Dict[str, Any]], filename: str) -> Path:
    """Render the Stage 5 dual-axis Pareto plots.

    Args:
        results: Ignored. Stage 5 loads :data:`config.METRICS_JSON`.
        filename: Ignored. Output stems are fixed inside the new module.

    Returns:
        Absolute path to the params Pareto figure.
    """

    if results or filename:
        logger.debug("save_pareto: legacy args ignored; reading %s", config.METRICS_JSON)
    with config.METRICS_JSON.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    paths = pareto.save_pareto(metrics)
    return paths[0]


def save_tsne(embeddings: Any, labels: List[str], filename: str) -> Path:
    """Render the Stage 5 2-panel low-dim representation figure.

    Args:
        embeddings: Ignored. The new module builds pooled span-saliency
            embeddings from cached ``predictions.npz`` files.
        labels: Ignored.
        filename: Ignored.

    Returns:
        Absolute path to the low-dim figure.
    """

    if embeddings is not None or labels or filename:
        logger.debug("save_tsne: legacy args ignored; reading %s", config.RUNS_DIR)
    path, _ = lowdim.save_lowdim()
    return path

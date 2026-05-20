"""Per-question-type F1 heatmap across variants and baselines.

Reads :data:`config.METRICS_JSON` and emits a single figure whose rows
are the 9 experimental variants plus the two no-train baselines and
whose columns are the question-type buckets defined in
:data:`config.QUESTION_TYPES`. Each cell shows the seed-mean F1 plus the
half-width of its 95%% CI.
"""

from __future__ import annotations

from src.plotting import _backend  # pylint: disable=unused-import  # backend pin

import logging
from pathlib import Path
from typing import Any, List, Mapping

import matplotlib.pyplot as plt
import numpy as np

import config

logger = logging.getLogger(__name__)

_FILENAME = "qtype_heatmap"


def _row_for(
    block: Mapping[str, Any], qtypes: List[str]
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(len(qtypes), dtype=float)
    halves = np.zeros(len(qtypes), dtype=float)
    per_qt = block.get("per_question_type", {})
    for i, qtype in enumerate(qtypes):
        entry = per_qt.get(qtype)
        if entry is None:
            continue
        f1 = entry.get("f1", {})
        means[i] = float(f1.get("mean") or 0.0)
        hw = f1.get("ci_half_width")
        halves[i] = float(hw) if hw is not None else 0.0
    return means, halves


def _save(fig: plt.Figure, stem: str) -> Path:
    out_dir = config.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = out_dir / f"{stem}.{config.PLOT_FORMAT}"
    fig.savefig(primary, dpi=config.PLOT_DPI, bbox_inches="tight")
    if config.PLOT_FORMAT != "png":
        fig.savefig(out_dir / f"{stem}.png", dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return primary


def save_qtype_heatmap(metrics: Mapping[str, Any]) -> Path:
    """Render the variants-x-qtypes F1 heatmap.

    Args:
        metrics: Parsed contents of :data:`config.METRICS_JSON`.

    Returns:
        Path to the vector figure (a PNG companion is also written).
    """

    variants = metrics.get("variants", {})
    baselines = metrics.get("baselines", {})

    row_names: List[str] = []
    blocks: List[Mapping[str, Any]] = []
    for name in config.VARIANTS:
        if name in variants:
            row_names.append(name)
            blocks.append(variants[name])
    for name in ("random_span", "tfidf"):
        if name in baselines:
            row_names.append(name)
            blocks.append(baselines[name])

    qtypes = list(config.QUESTION_TYPES)
    matrix = np.zeros((len(row_names), len(qtypes)), dtype=float)
    halves = np.zeros_like(matrix)
    for r, block in enumerate(blocks):
        matrix[r], halves[r] = _row_for(block, qtypes)

    column_order = np.argsort(-matrix.mean(axis=0))
    qtypes_ordered = [qtypes[i] for i in column_order]
    matrix = matrix[:, column_order]
    halves = halves[:, column_order]

    vcenter = float(np.median(matrix))
    vmin, vmax = float(matrix.min()), float(matrix.max())
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    half_range = max(vcenter - vmin, vmax - vcenter)

    fig, ax = plt.subplots(figsize=(1.1 * len(qtypes_ordered) + 2, 0.42 * len(row_names) + 1.6))
    image = ax.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=vcenter - half_range,
        vmax=vcenter + half_range,
        aspect="auto",
    )
    ax.set_xticks(range(len(qtypes_ordered)))
    ax.set_xticklabels(qtypes_ordered, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=9)
    ax.set_xlabel("question type")
    ax.set_title("F1 by variant x question type (mean +/- 95% CI half-width)")

    threshold = vcenter
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            mean = matrix[r, c]
            half = halves[r, c]
            colour = "white" if abs(mean - threshold) > half_range * 0.55 else "black"
            ax.text(
                c,
                r,
                f"{mean:.1f}\n\u00b1{half:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color=colour,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("F1")
    fig.tight_layout()
    path = _save(fig, _FILENAME)
    logger.info("wrote question-type heatmap to %s", path)
    return path


__all__ = ["save_qtype_heatmap"]

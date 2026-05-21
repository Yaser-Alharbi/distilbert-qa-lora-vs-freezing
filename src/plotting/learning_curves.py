"""Stage 5 learning-curve plots from cached ``history.json`` files.

Two figures are produced:

* ``learning_curves_grid``: a 3x3 small-multiples grid (one panel per
  experimental variant). Each panel shows seed-mean train and val loss
  vs ``global_step`` with shaded +/-1 std bands across the three seeds
  and vertical dashed lines at the epoch boundaries.
* ``learning_curves_val_overlay``: an overlay of every variant's
  seed-mean ``val_loss`` curve on shared axes (freezing branch solid,
  LoRA branch dashed).

The x-axis is taken straight from ``history["steps"]["global_step"]``
because ``history["meta"]["eval_every_steps"]`` reports 1263 in the
Stage 3 contract (the per-epoch summary cadence, not the step-level
logging cadence of 200). Inferring spacing from ``meta`` would mis-align
every plot.
"""

from __future__ import annotations

from src.plotting import _backend  # pylint: disable=unused-import  # backend pin

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config

logger = logging.getLogger(__name__)

_GRID_FILENAME = "learning_curves_grid"
_OVERLAY_FILENAME = "learning_curves_val_overlay"


def _load_history(variant: str, seed: int, runs_dir: Path) -> Dict[str, Any] | None:
    """Read one ``history.json``; return ``None`` if absent."""

    path = runs_dir / f"{variant}_seed{seed}" / "history.json"
    if not path.is_file():
        logger.warning("missing history.json for %s seed=%d", variant, seed)
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stack_curves(
    histories: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Return aligned (steps, train_mean+/-std, val_mean+/-std, epoch_boundaries).

    Seeds are aligned on the intersection of their step grids. In
    practice the three seeds for a variant share identical grids (the
    Stage 3 contract pins ``steps_per_epoch`` from a fixed-size training
    subset), so the intersection is the common grid.
    """

    grids = [tuple(int(s) for s in h["steps"]["global_step"]) for h in histories]
    common = sorted(set(grids[0]).intersection(*grids[1:]))
    steps = np.asarray(common, dtype=float)

    def _picks(history: Dict[str, Any], key: str) -> np.ndarray:
        idx = {int(s): i for i, s in enumerate(history["steps"]["global_step"])}
        return np.asarray(
            [history["steps"][key][idx[int(s)]] for s in common], dtype=float
        )

    train = np.stack([_picks(h, "train_loss") for h in histories], axis=0)
    val = np.stack([_picks(h, "val_loss") for h in histories], axis=0)

    boundaries = sorted(
        {int(b["global_step"]) for h in histories for b in h["epoch_boundaries"]}
    )
    return steps, train, val, boundaries


def _plot_panel(
    ax: plt.Axes,
    variant: str,
    steps: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    boundaries: List[int],
) -> None:
    train_mean = train.mean(axis=0)
    train_std = train.std(axis=0, ddof=0)
    val_mean = val.mean(axis=0)
    val_std = val.std(axis=0, ddof=0)

    ax.plot(steps, train_mean, color="tab:blue", label="train", linewidth=1.2)
    ax.fill_between(
        steps,
        train_mean - train_std,
        train_mean + train_std,
        color="tab:blue",
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(steps, val_mean, color="tab:orange", label="val", linewidth=1.2)
    ax.fill_between(
        steps,
        val_mean - val_std,
        val_mean + val_std,
        color="tab:orange",
        alpha=0.18,
        linewidth=0,
    )

    for boundary in boundaries:
        ax.axvline(boundary, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)

    ax.set_title(f"{variant}  val={val_mean[-1]:.2f}", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)


def _save(fig: plt.Figure, stem: str) -> List[Path]:
    """Write ``stem`` as the configured vector format plus a PNG companion."""

    out_dir = config.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = out_dir / f"{stem}.{config.PLOT_FORMAT}"
    fig.savefig(primary, dpi=config.PLOT_DPI, bbox_inches="tight")
    paths = [primary]
    if config.PLOT_FORMAT != "png":
        png = out_dir / f"{stem}.png"
        fig.savefig(png, dpi=config.PLOT_DPI, bbox_inches="tight")
        paths.append(png)
    plt.close(fig)
    return paths


def save_learning_curves(runs_dir: Path | None = None) -> List[Path]:
    """Render the small-multiples grid and the val-loss overlay.

    Args:
        runs_dir: Root directory containing ``{variant}_seed{seed}``
            subfolders. Defaults to :data:`config.RUNS_DIR`.

    Returns:
        Absolute paths to every figure file written (PDF + PNG when the
        configured format is vector).
    """

    runs_dir = runs_dir or config.RUNS_DIR
    variants = list(config.VARIANTS)
    per_variant: Dict[str, Dict[str, Any]] = {}

    for variant in variants:
        histories = [
            h
            for seed in config.SEEDS
            if (h := _load_history(variant, int(seed), runs_dir)) is not None
        ]
        if not histories:
            logger.warning("no history.json for variant %s; skipping panel", variant)
            continue
        steps, train, val, boundaries = _stack_curves(histories)
        per_variant[variant] = {
            "steps": steps,
            "train": train,
            "val": val,
            "boundaries": boundaries,
        }

    if not per_variant:
        logger.error("no learning curves to plot")
        return []

    n = len(variants)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.4 * cols, 2.4 * rows), sharex=False, sharey=False
    )
    axes_flat = np.asarray(axes).reshape(-1)
    for idx, variant in enumerate(variants):
        ax = axes_flat[idx]
        if variant not in per_variant:
            ax.set_visible(False)
            continue
        data = per_variant[variant]
        _plot_panel(
            ax, variant, data["steps"], data["train"], data["val"], data["boundaries"]
        )
        if idx % cols == 0:
            ax.set_ylabel("loss")
        if idx >= n - cols:
            ax.set_xlabel("global step")
    for spare in axes_flat[n:]:
        spare.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(
        "Stage 3 training curves (seed-mean +/- 1 std)", fontsize=11, y=1.0
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    paths = _save(fig, _GRID_FILENAME)
    logger.info("wrote learning-curves grid to %s", paths[0])

    overlay_fig, overlay_ax = plt.subplots(figsize=(7.0, 4.2))
    freezing = set(config.FREEZE_CONFIGS)
    cmap = plt.get_cmap("tab10")
    for idx, variant in enumerate(variants):
        if variant not in per_variant:
            continue
        data = per_variant[variant]
        val_mean = data["val"].mean(axis=0)
        linestyle = "-" if variant in freezing else "--"
        overlay_ax.plot(
            data["steps"],
            val_mean,
            label=variant,
            color=cmap(idx % 10),
            linewidth=1.3,
            linestyle=linestyle,
        )
    overlay_ax.set_xlabel("global step")
    overlay_ax.set_ylabel("val loss (seed mean)")
    overlay_ax.set_title("Validation loss across all 9 variants")
    overlay_ax.grid(alpha=0.25, linewidth=0.5)
    overlay_ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    overlay_paths = _save(overlay_fig, _OVERLAY_FILENAME)
    logger.info("wrote val-loss overlay to %s", overlay_paths[0])

    return paths + overlay_paths


__all__ = ["save_learning_curves"]

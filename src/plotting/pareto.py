"""Stage 5 complexity-vs-accuracy Pareto plots.

Two figures are produced from the cached Stage 4 ``metrics.json``:

* ``pareto_params``: F1 vs percentage of trainable parameters (log
  x-axis) with 95%% Student-t CI error bars; layer-freezing and LoRA
  shown as distinct marker shapes; the upper Pareto frontier drawn as a
  stepped line; matched-budget pairs annotated with brackets and the
  delta F1.
* ``pareto_walltime``: same y-axis against training wall-clock seconds
  (linear). Captures the compute-cost trade-off the params figure
  cannot.

Both figures carry horizontal reference lines for the two no-train
baselines and a dotted reference at the C3 (full fine-tune) F1.
"""

from __future__ import annotations

from src.plotting import _backend  # pylint: disable=unused-import  # backend pin

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config

logger = logging.getLogger(__name__)

_PARAMS_FILENAME = "pareto_params"
_WALLTIME_FILENAME = "pareto_walltime"


def _variant_kind(name: str) -> str:
    return "freeze" if name in config.FREEZE_CONFIGS else "lora"


def _collect_points(
    variants: Mapping[str, Dict[str, Any]],
    x_key: str,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for name, block in variants.items():
        f1 = block["f1"]
        x_block = block[x_key]
        x_value = x_block["mean"] if isinstance(x_block, dict) else x_block
        if x_value is None:
            continue
        points.append(
            {
                "name": name,
                "kind": _variant_kind(name),
                "x": float(x_value),
                "y": float(f1["mean"]),
                "y_low": _safe_float(f1.get("ci_low"), f1["mean"]),
                "y_high": _safe_float(f1.get("ci_high"), f1["mean"]),
            }
        )
    return points


def _safe_float(value: Any, fallback: float) -> float:
    return float(value) if value is not None else float(fallback)


def _pareto_frontier(points: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Return the (x, y) coords of the upper Pareto frontier (minimise x,
    maximise y), sorted by x ascending and y monotonically non-decreasing."""

    if not points:
        return []
    ordered = sorted(points, key=lambda p: (p["x"], -p["y"]))
    frontier: List[Tuple[float, float]] = []
    best_y = -np.inf
    for p in ordered:
        if p["y"] > best_y:
            frontier.append((p["x"], p["y"]))
            best_y = p["y"]
    return frontier


def _draw_scatter(
    ax: plt.Axes,
    points: Sequence[Dict[str, Any]],
    *,
    log_x: bool,
) -> None:
    style = {
        "freeze": {"marker": "o", "color": "tab:blue", "label": "freezing (C0-C3)"},
        "lora": {"marker": "s", "color": "tab:red", "label": "LoRA (r16-r256)"},
    }
    seen_kinds: set[str] = set()
    for point in points:
        kind = point["kind"]
        spec = style[kind]
        err = np.array(
            [[point["y"] - point["y_low"]], [point["y_high"] - point["y"]]]
        )
        ax.errorbar(
            point["x"],
            point["y"],
            yerr=err,
            fmt=spec["marker"],
            color=spec["color"],
            ecolor=spec["color"],
            elinewidth=0.9,
            capsize=2.5,
            markersize=6,
            label=spec["label"] if kind not in seen_kinds else None,
        )
        seen_kinds.add(kind)
        ax.annotate(
            point["name"],
            xy=(point["x"], point["y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    if log_x:
        ax.set_xscale("log")


def _draw_reference_lines(
    ax: plt.Axes,
    baselines: Mapping[str, Dict[str, Any]],
    c3_block: Dict[str, Any] | None,
) -> None:
    for name, label in (("random_span", "random_span F1"), ("tfidf", "tfidf F1")):
        block = baselines.get(name)
        if block is None:
            continue
        y = float(block["f1"]["mean"])
        ax.axhline(y, color="grey", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.annotate(
            label,
            xy=(ax.get_xlim()[1], y),
            xytext=(-4, 2),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="grey",
        )
    if c3_block is not None:
        y = float(c3_block["f1"]["mean"])
        ax.axhline(y, color="black", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.annotate(
            "C3 (full-FT)",
            xy=(ax.get_xlim()[1], y),
            xytext=(-4, 2),
            textcoords="offset points",
            ha="right",
            fontsize=7,
        )


def _draw_matched_budget(
    ax: plt.Axes,
    matched: Sequence[Dict[str, Any]],
    by_name: Mapping[str, Dict[str, Any]],
) -> None:
    for pair in matched:
        a_name, b_name = pair["a"], pair["b"]
        if a_name not in by_name or b_name not in by_name:
            continue
        a = by_name[a_name]
        b = by_name[b_name]
        xs = [a["x"], b["x"]]
        ys = [a["y"], b["y"]]
        ax.plot(xs, ys, color="black", linewidth=0.8, alpha=0.5)
        mid_x = float(np.exp(np.mean(np.log(xs)))) if min(xs) > 0 else float(np.mean(xs))
        mid_y = float(np.mean(ys))
        delta = float(pair["delta_f1"])
        ax.annotate(
            f"{a_name} vs {b_name}\n\u0394={delta:+.1f}",
            xy=(mid_x, mid_y),
            xytext=(6, -10),
            textcoords="offset points",
            fontsize=7,
            color="black",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "grey", "lw": 0.5, "alpha": 0.85},
        )


def _draw_frontier(ax: plt.Axes, frontier: Sequence[Tuple[float, float]]) -> None:
    if len(frontier) < 2:
        return
    xs = [p[0] for p in frontier]
    ys = [p[1] for p in frontier]
    ax.step(xs, ys, where="post", color="black", linewidth=0.9, alpha=0.6, label="Pareto frontier")


def _save(fig: plt.Figure, stem: str) -> List[Path]:
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


def save_pareto(metrics: Mapping[str, Any]) -> List[Path]:
    """Render the params and wall-clock Pareto plots.

    Args:
        metrics: Parsed contents of :data:`config.METRICS_JSON`.

    Returns:
        Absolute paths to every figure file written.
    """

    variants = metrics.get("variants", {})
    baselines = metrics.get("baselines", {})
    matched = metrics.get("matched_budget", [])
    c3_block = variants.get("C3")

    paths: List[Path] = []

    params_points = _collect_points(variants, "pct_trainable")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    _draw_scatter(ax, params_points, log_x=True)
    by_name = {p["name"]: p for p in params_points}
    _draw_frontier(ax, _pareto_frontier(params_points))
    ax.set_xlabel("% trainable parameters (log)")
    ax.set_ylabel("F1 (95% CI)")
    ax.set_title("Complexity vs accuracy")
    ax.grid(alpha=0.25, linewidth=0.5, which="both")
    _draw_reference_lines(ax, baselines, c3_block)
    _draw_matched_budget(ax, matched, by_name)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    params_paths = _save(fig, _PARAMS_FILENAME)
    logger.info("wrote params Pareto to %s", params_paths[0])
    paths.extend(params_paths)

    walltime_points = _collect_points(variants, "train_wall_clock_sec")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    _draw_scatter(ax, walltime_points, log_x=False)
    _draw_frontier(ax, _pareto_frontier(walltime_points))
    ax.set_xlabel("training wall-clock (seconds)")
    ax.set_ylabel("F1 (95% CI)")
    ax.set_title("Compute cost vs accuracy")
    ax.grid(alpha=0.25, linewidth=0.5)
    _draw_reference_lines(ax, baselines, c3_block)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    walltime_paths = _save(fig, _WALLTIME_FILENAME)
    logger.info("wrote walltime Pareto to %s", walltime_paths[0])
    paths.extend(walltime_paths)

    return paths


__all__ = ["save_pareto"]

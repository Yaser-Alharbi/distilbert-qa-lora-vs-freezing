"""Consolidate Stage 5 analysis artefacts into a single JSON file.

Bridges the figures and Tier B outputs into the headline numbers
referenced by the report's Results/Conclusion section.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import config

logger = logging.getLogger(__name__)


def _safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return float(value)
    return value


def _headline_ranking(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, block in metrics.get("variants", {}).items():
        f1 = block.get("f1", {})
        rows.append(
            {
                "name": name,
                "kind": "variant",
                "f1_mean": _safe(f1.get("mean")),
                "f1_ci_low": _safe(f1.get("ci_low")),
                "f1_ci_high": _safe(f1.get("ci_high")),
                "pct_trainable": _safe(block.get("pct_trainable")),
            }
        )
    for name, block in metrics.get("baselines", {}).items():
        f1 = block.get("f1", {})
        rows.append(
            {
                "name": name,
                "kind": "baseline",
                "f1_mean": _safe(f1.get("mean")),
                "f1_ci_low": _safe(f1.get("ci_low")),
                "f1_ci_high": _safe(f1.get("ci_high")),
                "pct_trainable": 0.0,
            }
        )
    rows.sort(key=lambda r: (r["f1_mean"] is None, -(r["f1_mean"] or 0.0)))
    return rows


def _matched_budget_block(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    pairs_out: List[Dict[str, Any]] = []
    wins = 0
    losses = 0
    for pair in metrics.get("matched_budget", []):
        delta = float(pair.get("delta_f1", 0.0))
        beats = delta > 0
        if beats:
            wins += 1
        elif delta < 0:
            losses += 1
        pairs_out.append({**pair, "lora_beats_freezing": beats})
    if pairs_out and wins == len(pairs_out):
        verdict = "LoRA beats freezing at matched budget"
    elif pairs_out and losses == len(pairs_out):
        verdict = "LoRA loses to freezing at matched budget"
    elif pairs_out:
        verdict = "mixed"
    else:
        verdict = "n/a"
    return {"pairs": pairs_out, "hypothesis_verdict": verdict}


def _per_qtype_best_worst(metrics: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    variants = metrics.get("variants", {})
    for qtype in config.QUESTION_TYPES:
        best_name: Optional[str] = None
        worst_name: Optional[str] = None
        best_f1 = -float("inf")
        worst_f1 = float("inf")
        for name, block in variants.items():
            entry = block.get("per_question_type", {}).get(qtype)
            if entry is None:
                continue
            mean = entry.get("f1", {}).get("mean")
            if mean is None:
                continue
            if mean > best_f1:
                best_f1 = mean
                best_name = name
            if mean < worst_f1:
                worst_f1 = mean
                worst_name = name
        rows[qtype] = {
            "best": {"variant": best_name, "f1_mean": None if best_name is None else float(best_f1)},
            "worst": {"variant": worst_name, "f1_mean": None if worst_name is None else float(worst_f1)},
        }
    return rows


def _convergence_block(runs_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    c3_gap: Optional[float] = None
    for variant in config.VARIANTS:
        train_vals: List[float] = []
        val_vals: List[float] = []
        for seed in config.SEEDS:
            history_path = runs_dir / f"{variant}_seed{seed}" / "history.json"
            if not history_path.is_file():
                continue
            with history_path.open("r", encoding="utf-8") as handle:
                history = json.load(handle)
            train_loss = history.get("epoch_train_loss") or []
            val_loss = history.get("epoch_val_loss") or []
            if train_loss:
                train_vals.append(float(train_loss[-1]))
            if val_loss:
                val_vals.append(float(val_loss[-1]))
        if not train_vals or not val_vals:
            continue
        mean_train = sum(train_vals) / len(train_vals)
        mean_val = sum(val_vals) / len(val_vals)
        out[variant] = {
            "final_train_loss_mean": float(mean_train),
            "final_val_loss_mean": float(mean_val),
            "val_minus_train": float(mean_val - mean_train),
        }
        if variant == "C3":
            c3_gap = float(mean_val - mean_train)
    return {"per_variant": out, "c3_overfit_gap": c3_gap}


def write_summary(
    metrics: Mapping[str, Any],
    plot_paths: List[Path],
    lowdim_silhouettes: Optional[Dict[str, float]],
    calibration_ece: Optional[float],
    tier_b_generated: bool,
    runs_dir: Path | None = None,
) -> Path:
    """Aggregate Stage 5 outputs into :data:`config.ANALYSIS_SUMMARY_JSON`.

    Args:
        metrics: Parsed :data:`config.METRICS_JSON` payload.
        plot_paths: Absolute paths of every figure file written.
        lowdim_silhouettes: Per-variant silhouette scores from the
            low-dim panel, or ``None`` when Tier B was skipped.
        calibration_ece: Expected Calibration Error from the
            reliability diagram, or ``None`` when Tier B was skipped.
        tier_b_generated: Whether the Tier B path ran.
        runs_dir: Override for the per-run history root. Defaults to
            :data:`config.RUNS_DIR`.

    Returns:
        Path to the written JSON summary.
    """

    runs_dir = runs_dir or config.RUNS_DIR
    project_root = config.ROOT_DIR
    relative_plots = []
    for path in plot_paths:
        try:
            relative_plots.append(str(path.relative_to(project_root)))
        except ValueError:
            relative_plots.append(str(path))

    payload: Dict[str, Any] = {
        "headline_ranking": _headline_ranking(metrics),
        "matched_budget": _matched_budget_block(metrics),
        "per_qtype_best_worst": _per_qtype_best_worst(metrics),
        "convergence": _convergence_block(runs_dir),
        "lowdim_silhouettes": (
            {k: float(v) for k, v in lowdim_silhouettes.items()}
            if lowdim_silhouettes
            else None
        ),
        "calibration_ece": float(calibration_ece) if calibration_ece is not None else None,
        "plots": relative_plots,
        "tier_b_generated": bool(tier_b_generated),
    }

    config.ANALYSIS_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with config.ANALYSIS_SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    logger.info("wrote analysis summary to %s", config.ANALYSIS_SUMMARY_JSON)
    return config.ANALYSIS_SUMMARY_JSON


__all__ = ["write_summary"]

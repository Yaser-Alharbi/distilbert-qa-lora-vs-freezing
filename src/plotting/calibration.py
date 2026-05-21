"""Reliability diagram for a single high-capacity variant (Tier B).

Decodes the best valid span per example (HF n-best constraints) and
attaches a confidence score:

  ``conf = p_start[s_best] * p_end[e_best]``

where ``p_start`` and ``p_end`` are softmax distributions restricted to
context-mask positions on the chosen feature window. Confidences are
binned into equal-mass quantile bins; the reliability diagram plots
mean bin confidence against the empirical rate of correct predictions
(F1 >= 0.5 against the gold answers).
"""

from __future__ import annotations

from src.plotting import _backend  # pylint: disable=unused-import  # backend pin

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config
from src.data.loader import validation_example_index
from src.evaluation.metrics import compute_em_f1

logger = logging.getLogger(__name__)

_FILENAME_PREFIX = "calibration_"
_CORRECT_THRESHOLD_F1 = 50.0
_MASK_PENALTY = -1.0e9


def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    biased = np.where(mask, logits, _MASK_PENALTY).astype(np.float64)
    biased -= biased.max()
    probs = np.exp(biased)
    total = probs.sum()
    if total <= 0:
        return np.zeros_like(probs)
    return probs / total


def _best_span_per_example(
    npz_path: Path, id_to_context: Dict[str, str]
) -> Dict[str, Tuple[str, float]]:
    """Return ``{example_id: (prediction_text, confidence)}``.

    Mirrors :func:`src.evaluation.postprocess.decode_predictions`'
    candidate scan but additionally records the joint softmax confidence
    at the chosen ``(s, e)`` positions of the chosen feature.
    """

    with np.load(npz_path, allow_pickle=False) as handle:
        start_logits = handle["start_logits"]
        end_logits = handle["end_logits"]
        example_ids = handle["example_ids"]
        offsets = handle["offsets"]
        context_mask = handle["context_mask"]

    feature_groups: Dict[str, List[int]] = defaultdict(list)
    for row, example_id in enumerate(example_ids):
        feature_groups[str(example_id)].append(row)

    n_best = int(config.N_BEST_SIZE)
    max_len = int(config.MAX_ANSWER_LENGTH)
    out: Dict[str, Tuple[str, float]] = {}

    for example_id, rows in feature_groups.items():
        best: Tuple[float, int, int, int] | None = None
        for row in rows:
            sl = start_logits[row]
            el = end_logits[row]
            cm = context_mask[row].astype(bool)
            off = offsets[row]

            start_top = np.argsort(-sl)[:n_best]
            end_top = np.argsort(-el)[:n_best]
            start_valid = cm[start_top] & ((off[start_top, 0] != 0) | (off[start_top, 1] != 0))
            end_valid = cm[end_top] & ((off[end_top, 0] != 0) | (off[end_top, 1] != 0))
            start_top = start_top[start_valid]
            end_top = end_top[end_valid]
            if start_top.size == 0 or end_top.size == 0:
                continue
            starts = start_top[:, None]
            ends = end_top[None, :]
            length_ok = (ends >= starts) & ((ends - starts + 1) <= max_len)
            if not length_ok.any():
                continue
            scores = sl[starts] + el[ends]
            scores = np.where(length_ok, scores, -np.inf)
            flat = int(np.argmax(scores))
            s_pos, e_pos = np.unravel_index(flat, scores.shape)
            score = float(scores[s_pos, e_pos])
            if not np.isfinite(score):
                continue
            s_idx = int(starts[s_pos, 0])
            e_idx = int(ends[0, e_pos])
            if best is None or score > best[0]:
                best = (score, row, s_idx, e_idx)

        if best is None:
            out[example_id] = ("", 0.0)
            continue
        _, row, s_idx, e_idx = best
        context = id_to_context.get(example_id)
        if context is None:
            raise KeyError(example_id)
        char_start = int(offsets[row, s_idx, 0])
        char_end = int(offsets[row, e_idx, 1])
        prediction = context[char_start:char_end]
        p_start = _softmax_masked(start_logits[row], context_mask[row].astype(bool))
        p_end = _softmax_masked(end_logits[row], context_mask[row].astype(bool))
        confidence = float(p_start[s_idx] * p_end[e_idx])
        out[example_id] = (prediction, confidence)
    return out


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    edges = np.unique(edges)
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)


def _save(fig: plt.Figure, stem: str) -> Path:
    out_dir = config.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = out_dir / f"{stem}.{config.PLOT_FORMAT}"
    fig.savefig(primary, dpi=config.PLOT_DPI, bbox_inches="tight")
    if config.PLOT_FORMAT != "png":
        fig.savefig(out_dir / f"{stem}.png", dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return primary


def save_calibration(
    variant: str | None = None,
    seed: int | None = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Render the reliability diagram and return calibration metrics.

    Args:
        variant: Variant name. Defaults to :data:`config.CALIBRATION_VARIANT`.
        seed: Seed within the variant. Defaults to :data:`config.SEEDS[0]`.

    Returns:
        ``(figure_path, {"ece": float, "n": int, "bins": [...]})``.
    """

    variant = variant or config.CALIBRATION_VARIANT
    seed = int(seed if seed is not None else config.SEEDS[0])
    npz_path = config.RUNS_DIR / f"{variant}_seed{seed}" / "predictions.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    ordered_ids, id_to_context, _, id_to_golds, _ = validation_example_index()
    decoded = _best_span_per_example(npz_path, id_to_context)

    confidences: List[float] = []
    correct: List[float] = []
    for example_id in ordered_ids:
        prediction, conf = decoded.get(example_id, ("", 0.0))
        f1 = compute_em_f1([prediction], [id_to_golds[example_id]])["f1"]
        confidences.append(conf)
        correct.append(1.0 if f1 >= _CORRECT_THRESHOLD_F1 else 0.0)
    conf_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)

    bin_idx = _quantile_bins(conf_arr, int(config.CALIBRATION_BINS))
    bins: List[Dict[str, float]] = []
    ece = 0.0
    n_total = conf_arr.shape[0]
    for b in range(int(bin_idx.max()) + 1):
        sel = bin_idx == b
        if not sel.any():
            continue
        mean_conf = float(conf_arr[sel].mean())
        emp = float(correct_arr[sel].mean())
        count = int(sel.sum())
        ece += (count / n_total) * abs(mean_conf - emp)
        bins.append({"bin": b, "n": count, "mean_confidence": mean_conf, "empirical_correct": emp})

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(5.2, 5.4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    xs = np.array([b["mean_confidence"] for b in bins])
    ys = np.array([b["empirical_correct"] for b in bins])
    counts = np.array([b["n"] for b in bins])
    ax_rel.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8, label="perfect calibration")
    ax_rel.plot(xs, ys, marker="o", color="tab:blue", linewidth=1.2, label="empirical")
    ax_rel.set_ylim(-0.02, 1.02)
    ax_rel.set_ylabel("empirical P(F1 >= 0.5)")
    ax_rel.set_title(f"{variant} reliability (ECE={ece:.3f}, n={n_total})")
    ax_rel.grid(alpha=0.25, linewidth=0.5)
    ax_rel.legend(loc="upper left", fontsize=8, frameon=False)

    ax_hist.bar(xs, counts, width=0.05, color="tab:blue", alpha=0.6)
    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.set_xlabel("joint best-span confidence (mean per bin)")
    ax_hist.set_ylabel("count")
    fig.tight_layout()
    path = _save(fig, f"{_FILENAME_PREFIX}{variant}")
    logger.info("wrote calibration plot for %s to %s (ECE=%.3f)", variant, path, ece)
    return path, {"ece": float(ece), "n": int(n_total), "bins": bins, "variant": variant, "seed": seed}


__all__ = ["save_calibration"]

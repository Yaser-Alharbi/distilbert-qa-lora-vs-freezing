"""2-D low-dim representation analysis (Tier B).

Compares a low-capacity variant against a high-capacity variant by
projecting a pooled summary of their span-saliency logits to 2-D and
colouring the points by question type. No model is reloaded; we treat
the cached ``start_logits`` and ``end_logits`` arrays as the learned
representation that distinguishes examples.

Embedding construction (per example)
------------------------------------

1. Collect every feature window for the example.
2. Restrict ``start_logits`` and ``end_logits`` to context-mask
   positions on each feature; score the feature by
   ``max(start) + max(end)`` and keep the feature with the highest
   score (the same feature ``decode_predictions`` would prefer).
3. From the chosen feature's start and end logit vectors (context
   positions only) compute eight scalars each:

   * max, mean, std
   * Shannon entropy of the softmax over context positions
   * argmax position normalised to ``[0, 1]``
   * top-3 logit gaps ``l_0 - l_k`` for ``k in {1, 2, 3}``.

   Concatenate to get a 16-D vector per example.

This is a cheap statistical fingerprint of the model's span-saliency
distribution; a more capable model should produce sharper, more
discriminative fingerprints per question type.
"""

from __future__ import annotations

from src.plotting import _backend  # pylint: disable=unused-import  # backend pin

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import config
from src.data.loader import validation_example_index

logger = logging.getLogger(__name__)

_FILENAME = "lowdim_2panel"
_EMBED_DIM = 16
_TOPK_GAPS = 3
_MIN_TSNE_POINTS = 50


def _shannon_entropy(logits: np.ndarray) -> float:
    if logits.size == 0:
        return 0.0
    log_probs = logits - logits.max()
    probs = np.exp(log_probs)
    probs /= probs.sum()
    nonzero = probs[probs > 0]
    return float(-(nonzero * np.log(nonzero)).sum())


def _topk_gaps(logits: np.ndarray, k: int) -> np.ndarray:
    if logits.size <= k:
        padded = np.concatenate([logits, np.full(k + 1 - logits.size, logits.min())])
        sorted_desc = np.sort(padded)[::-1]
    else:
        sorted_desc = np.sort(logits)[::-1][: k + 1]
    return sorted_desc[0] - sorted_desc[1 : k + 1]


def _summarise(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return np.zeros(_EMBED_DIM // 2, dtype=float)
    argmax_norm = float(np.argmax(logits)) / max(logits.size - 1, 1)
    return np.concatenate(
        [
            np.array([float(logits.max()), float(logits.mean()), float(logits.std())]),
            np.array([_shannon_entropy(logits)]),
            np.array([argmax_norm]),
            _topk_gaps(logits, _TOPK_GAPS),
        ]
    )


def _build_embeddings(
    npz_path: Path,
) -> Tuple[List[str], np.ndarray]:
    """Return ``(example_ids, embeddings)`` for every distinct example."""

    with np.load(npz_path, allow_pickle=False) as handle:
        start_logits = handle["start_logits"]
        end_logits = handle["end_logits"]
        example_ids = handle["example_ids"]
        context_mask = handle["context_mask"]

    feature_groups: Dict[str, List[int]] = {}
    for row, example_id in enumerate(example_ids):
        feature_groups.setdefault(str(example_id), []).append(row)

    ordered_ids: List[str] = list(feature_groups)
    embeddings = np.zeros((len(ordered_ids), _EMBED_DIM), dtype=float)

    for i, example_id in enumerate(ordered_ids):
        best_row: int | None = None
        best_score = -np.inf
        best_start: np.ndarray | None = None
        best_end: np.ndarray | None = None
        for row in feature_groups[example_id]:
            mask = context_mask[row].astype(bool)
            if not mask.any():
                continue
            start_ctx = start_logits[row][mask]
            end_ctx = end_logits[row][mask]
            score = float(start_ctx.max()) + float(end_ctx.max())
            if score > best_score:
                best_score = score
                best_row = row
                best_start = start_ctx
                best_end = end_ctx
        if best_row is None or best_start is None or best_end is None:
            continue
        embeddings[i] = np.concatenate(
            [_summarise(best_start), _summarise(best_end)]
        )
    return ordered_ids, embeddings


def _project(embeddings: np.ndarray, *, method: str) -> Tuple[np.ndarray, str]:
    """Project ``(n, _EMBED_DIM)`` to ``(n, 2)`` via t-SNE or PCA."""

    n = embeddings.shape[0]
    if method == "tsne" and n >= _MIN_TSNE_POINTS:
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("sklearn.manifold.TSNE unavailable; falling back to PCA")
        else:
            tsne = TSNE(
                n_components=2,
                perplexity=min(config.TSNE_PERPLEXITY, max((n - 1) // 3, 5)),
                init="pca",
                random_state=int(config.SEEDS[0]),
                learning_rate="auto",
            )
            return tsne.fit_transform(embeddings), "tsne"

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=int(config.SEEDS[0]))
    return pca.fit_transform(embeddings), "pca"


def _silhouette(coords: np.ndarray, labels: List[str]) -> float | None:
    if len(set(labels)) < 2 or coords.shape[0] < 3:
        return None
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None
    return float(silhouette_score(coords, labels))


def _draw_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    labels: List[str],
    title: str,
) -> None:
    qtypes = list(config.QUESTION_TYPES)
    cmap = plt.get_cmap("tab10")
    for idx, qtype in enumerate(qtypes):
        mask = np.array([lab == qtype for lab in labels], dtype=bool)
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.6,
            color=cmap(idx % 10),
            label=qtype,
            edgecolors="none",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _save(fig: plt.Figure, stem: str) -> Path:
    out_dir = config.PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = out_dir / f"{stem}.{config.PLOT_FORMAT}"
    fig.savefig(primary, dpi=config.PLOT_DPI, bbox_inches="tight")
    if config.PLOT_FORMAT != "png":
        fig.savefig(out_dir / f"{stem}.png", dpi=config.PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return primary


def save_lowdim(
    variant_low: str | None = None,
    variant_high: str | None = None,
    seed: int | None = None,
) -> Tuple[Path, Dict[str, float]]:
    """Render the 2-panel low-dim figure and return panel silhouettes.

    Args:
        variant_low: Low-capacity variant name. Defaults to
            :data:`config.LOWDIM_LOW_VARIANT`.
        variant_high: High-capacity variant name. Defaults to
            :data:`config.LOWDIM_HIGH_VARIANT`.
        seed: Seed to use for both variants. Defaults to
            :data:`config.SEEDS[0]`.

    Returns:
        ``(figure_path, silhouettes)`` where ``silhouettes`` maps each
        variant name to its 2-D silhouette score against question-type
        labels (``None`` when the projection is degenerate).
    """

    variant_low = variant_low or config.LOWDIM_LOW_VARIANT
    variant_high = variant_high or config.LOWDIM_HIGH_VARIANT
    seed = int(seed if seed is not None else config.SEEDS[0])

    _, _, _, _, id_to_qtype = validation_example_index()
    rng = np.random.default_rng(int(config.SEEDS[0]))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6))
    silhouettes: Dict[str, float] = {}

    for ax, variant in zip(axes, (variant_low, variant_high)):
        npz_path = config.RUNS_DIR / f"{variant}_seed{seed}" / "predictions.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        ids, embeddings = _build_embeddings(npz_path)
        n = embeddings.shape[0]
        if n > config.TSNE_MAX_POINTS:
            indices = rng.choice(n, size=config.TSNE_MAX_POINTS, replace=False)
            indices.sort()
            embeddings = embeddings[indices]
            ids = [ids[i] for i in indices]
        labels = [id_to_qtype.get(eid, "other") for eid in ids]
        coords, method = _project(embeddings, method=config.LOWDIM_METHOD)
        sil = _silhouette(coords, labels)
        if sil is not None:
            silhouettes[variant] = sil
        sil_str = f"silhouette={sil:.3f}" if sil is not None else "silhouette=n/a"
        _draw_panel(ax, coords, labels, f"{variant}  ({method.upper()}, {sil_str})")

    handles, labels_legend = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="lower center",
        ncol=min(len(labels_legend), 9),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Span-saliency embeddings projected to 2-D, coloured by question type")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path = _save(fig, _FILENAME)
    logger.info("wrote low-dim 2-panel to %s (silhouettes=%s)", path, silhouettes)
    return path, silhouettes


__all__ = ["save_lowdim"]

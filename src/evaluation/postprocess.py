"""HF-style n-best constrained span decoding from cached Stage 3 logits.

A single function, :func:`decode_predictions`, consumes the ``predictions.npz``
file written by ``src/training/trainer.py`` and emits the best-span string
prediction per validation ``example_id``. Decoding is purely numpy + Python
string slicing; no model is reloaded.

For every example we collect candidate ``(start, end)`` pairs across **all**
of its feature windows, score them by ``start_logits[s] + end_logits[e]``,
apply the standard HF constraints, and take the global argmax. Examples with
no valid candidate (very rare) get the empty string.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


def _best_span_for_feature(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    context_mask: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[float, int, int] | None:
    """Return the best ``(score, start_idx, end_idx)`` for one feature row.

    Implements the HF n-best constraint set: top-``N_BEST_SIZE`` start and
    end indices by logit, rejected unless both fall in the context segment
    and yield a span of valid length. ``None`` if no candidate survives.
    """

    n_best = int(config.N_BEST_SIZE)
    max_len = int(config.MAX_ANSWER_LENGTH)

    start_top = np.argsort(-start_logits)[:n_best]
    end_top = np.argsort(-end_logits)[:n_best]

    start_valid = (
        context_mask[start_top]
        & ((offsets[start_top, 0] != 0) | (offsets[start_top, 1] != 0))
    )
    end_valid = (
        context_mask[end_top]
        & ((offsets[end_top, 0] != 0) | (offsets[end_top, 1] != 0))
    )
    start_top = start_top[start_valid]
    end_top = end_top[end_valid]
    if start_top.size == 0 or end_top.size == 0:
        return None

    starts = start_top[:, None]
    ends = end_top[None, :]
    length_ok = (ends >= starts) & ((ends - starts + 1) <= max_len)
    if not length_ok.any():
        return None

    scores = start_logits[starts] + end_logits[ends]
    scores = np.where(length_ok, scores, -np.inf)
    flat_idx = int(np.argmax(scores))
    s_pos, e_pos = np.unravel_index(flat_idx, scores.shape)
    best_score = float(scores[s_pos, e_pos])
    if not np.isfinite(best_score):
        return None
    return best_score, int(starts[s_pos, 0]), int(ends[0, e_pos])


def decode_predictions(
    npz_path: Path, id_to_context: Mapping[str, str]
) -> Tuple[Dict[str, str], float]:
    """Decode all feature rows in ``npz_path`` into one prediction per example.

    Args:
        npz_path: Path to a Stage 3 ``predictions.npz`` file containing the
            keys ``start_logits``, ``end_logits``, ``example_ids``,
            ``offsets``, ``context_mask``.
        id_to_context: Mapping from validation ``example_id`` to its raw
            context string (used for the final character slice).

    Returns:
        ``(predictions, decode_ms_per_example)`` where ``predictions``
        maps every distinct ``example_id`` in the file to its best
        predicted answer (possibly ``""``), and the latency is the mean
        wall-clock time of the decode loop in milliseconds.
    """

    if not npz_path.is_file():
        raise FileNotFoundError(f"predictions file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as handle:
        start_logits = handle["start_logits"]
        end_logits = handle["end_logits"]
        example_ids = handle["example_ids"]
        offsets = handle["offsets"]
        context_mask = handle["context_mask"]

    n_features = start_logits.shape[0]
    if not (
        end_logits.shape[0] == n_features
        == example_ids.shape[0] == offsets.shape[0] == context_mask.shape[0]
    ):
        raise ValueError(
            f"predictions arrays have inconsistent leading dim in {npz_path}"
        )

    feature_groups: Dict[str, List[int]] = defaultdict(list)
    for row, example_id in enumerate(example_ids):
        feature_groups[str(example_id)].append(row)

    predictions: Dict[str, str] = {}
    t0 = time.perf_counter()
    for example_id, rows in feature_groups.items():
        context = id_to_context.get(example_id)
        if context is None:
            raise KeyError(
                f"example_id {example_id!r} from {npz_path} has no context "
                "in the validation index"
            )

        best: Tuple[float, int, int, int] | None = None
        for row in rows:
            candidate = _best_span_for_feature(
                start_logits[row],
                end_logits[row],
                context_mask[row],
                offsets[row],
            )
            if candidate is None:
                continue
            score, s_idx, e_idx = candidate
            if best is None or score > best[0]:
                best = (score, row, s_idx, e_idx)

        if best is None:
            predictions[example_id] = ""
            continue
        _, row, s_idx, e_idx = best
        char_start = int(offsets[row, s_idx, 0])
        char_end = int(offsets[row, e_idx, 1])
        predictions[example_id] = context[char_start:char_end]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    n_examples = len(feature_groups)
    decode_ms_per_example = elapsed_ms / n_examples if n_examples else 0.0
    logger.debug(
        "decoded %d examples (%d features) from %s in %.1f ms (%.3f ms/ex)",
        n_examples, n_features, npz_path.name, elapsed_ms, decode_ms_per_example,
    )
    return predictions, decode_ms_per_example

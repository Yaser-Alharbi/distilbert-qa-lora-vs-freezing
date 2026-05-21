"""Qualitative error analysis: surface ``N_ERROR_EXAMPLES`` failures.

For a high-capacity variant (default :data:`config.ERROR_VARIANT`,
:data:`config.SEEDS[0]`):

* decode predicted spans through :func:`decode_predictions`,
* compute per-example F1 against the gold answers,
* keep failures (F1 < 50) and auto-categorise each as one of
  ``boundary`` (token overlap with gold but wrong span),
  ``wrong_sentence`` (prediction comes from a different sentence than
  the gold), or ``distractor`` (no token overlap and a noun-phrase-like
  surface form), and
* select a stratified subset of `n` cases covering >=4 question types
  and >=1 case of each category when available.

Output is written to :data:`config.ERROR_CASES_JSON`.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import config
from src.data.loader import validation_example_index
from src.evaluation.metrics import compute_em_f1
from src.evaluation.postprocess import decode_predictions

logger = logging.getLogger(__name__)

_FAILURE_THRESHOLD_F1 = 50.0
_SNIPPET_WINDOW = 200
_MIN_QTYPES = 4
_CATEGORY_ORDER = ("boundary", "wrong_sentence", "distractor")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMERIC_TOKEN = re.compile(r"^\d+(?:[.,]\d+)*$")


def _tokenise(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", text.strip().lower()) if t]


def _sentence_indices(context: str) -> List[int]:
    """Return the start offsets of each sentence in ``context``."""

    starts: List[int] = [0]
    for match in _SENTENCE_SPLIT.finditer(context):
        starts.append(match.end())
    return starts


def _sentence_index_for(offset: int, starts: Sequence[int]) -> int:
    for i in range(len(starts) - 1, -1, -1):
        if starts[i] <= offset:
            return i
    return 0


def _categorise(
    prediction: str,
    golds: Sequence[str],
    context: str,
    f1: float,
) -> str:
    pred_tokens = set(_tokenise(prediction))
    gold_token_sets = [set(_tokenise(g)) for g in golds] or [set()]
    overlap = max((len(pred_tokens & g) for g in gold_token_sets), default=0)

    if overlap > 0 and f1 < _FAILURE_THRESHOLD_F1:
        return "boundary"

    if golds and prediction:
        sentence_starts = _sentence_indices(context)
        pred_offset = context.find(prediction)
        gold_offset = -1
        for gold in golds:
            idx = context.find(gold)
            if idx >= 0:
                gold_offset = idx
                break
        if pred_offset >= 0 and gold_offset >= 0:
            pred_sentence = _sentence_index_for(pred_offset, sentence_starts)
            gold_sentence = _sentence_index_for(gold_offset, sentence_starts)
            if pred_sentence != gold_sentence:
                return "wrong_sentence"

    if overlap == 0 and prediction:
        tokens = prediction.split()
        has_capitalised = any(tok[:1].isupper() for tok in tokens)
        has_numeric = any(_NUMERIC_TOKEN.match(tok) for tok in tokens)
        if has_capitalised or has_numeric:
            return "distractor"
    return "other"


def _context_snippet(context: str, golds: Sequence[str]) -> str:
    for gold in golds:
        idx = context.find(gold)
        if idx >= 0:
            start = max(0, idx - _SNIPPET_WINDOW)
            end = min(len(context), idx + len(gold) + _SNIPPET_WINDOW)
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(context) else ""
            return f"{prefix}{context[start:end]}{suffix}"
    return context[: 2 * _SNIPPET_WINDOW] + ("..." if len(context) > 2 * _SNIPPET_WINDOW else "")


def _select_stratified(
    candidates: List[Dict[str, Any]], n: int
) -> List[Dict[str, Any]]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in candidates:
        by_category[case["category"]].append(case)
    for bucket in by_category.values():
        bucket.sort(key=lambda c: c["f1"])

    chosen: List[Dict[str, Any]] = []
    seen_qtypes: set[str] = set()

    for category in _CATEGORY_ORDER:
        bucket = by_category.get(category, [])
        for case in bucket:
            if case in chosen:
                continue
            chosen.append(case)
            seen_qtypes.add(case["question_type"])
            break

    if len(seen_qtypes) < _MIN_QTYPES:
        remaining = sorted(
            (c for c in candidates if c not in chosen),
            key=lambda c: c["f1"],
        )
        for case in remaining:
            if case["question_type"] in seen_qtypes:
                continue
            chosen.append(case)
            seen_qtypes.add(case["question_type"])
            if len(seen_qtypes) >= _MIN_QTYPES or len(chosen) >= n:
                break

    if len(chosen) < n:
        remaining = sorted(
            (c for c in candidates if c not in chosen),
            key=lambda c: c["f1"],
        )
        chosen.extend(remaining[: n - len(chosen)])

    return chosen[:n]


def extract_error_cases(
    variant: str | None = None,
    seed: int | None = None,
    n: int | None = None,
) -> Path:
    """Write the stratified failure list to :data:`config.ERROR_CASES_JSON`.

    Args:
        variant: Variant whose predictions to inspect. Defaults to
            :data:`config.ERROR_VARIANT`.
        seed: Seed within the variant. Defaults to :data:`config.SEEDS[0]`.
        n: Number of cases to emit. Defaults to
            :data:`config.N_ERROR_EXAMPLES`.

    Returns:
        Path to the written JSON file.
    """

    variant = variant or config.ERROR_VARIANT
    seed = int(seed if seed is not None else config.SEEDS[0])
    n = int(n if n is not None else config.N_ERROR_EXAMPLES)

    npz_path = config.RUNS_DIR / f"{variant}_seed{seed}" / "predictions.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    ordered_ids, id_to_context, id_to_question, id_to_golds, id_to_qtype = (
        validation_example_index()
    )
    predictions, _ = decode_predictions(npz_path, id_to_context)

    candidates: List[Dict[str, Any]] = []
    for example_id in ordered_ids:
        prediction = predictions.get(example_id, "")
        golds = id_to_golds[example_id]
        f1 = compute_em_f1([prediction], [golds])["f1"]
        if f1 >= _FAILURE_THRESHOLD_F1:
            continue
        context = id_to_context[example_id]
        category = _categorise(prediction, golds, context, f1)
        candidates.append(
            {
                "id": example_id,
                "question": id_to_question[example_id],
                "context_snippet": _context_snippet(context, golds),
                "golds": list(golds),
                "prediction": prediction,
                "f1": float(f1),
                "question_type": id_to_qtype.get(example_id, "other"),
                "category": category,
            }
        )

    if not candidates:
        logger.warning("no failures found for %s seed=%d; writing empty case list", variant, seed)

    selected = _select_stratified(candidates, n)

    payload: Dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "n_failures": len(candidates),
        "n_emitted": len(selected),
        "cases": selected,
    }
    config.ERROR_CASES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with config.ERROR_CASES_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
    logger.info(
        "wrote %d error cases (from %d failures) to %s",
        len(selected),
        len(candidates),
        config.ERROR_CASES_JSON,
    )
    return config.ERROR_CASES_JSON


__all__ = ["extract_error_cases"]

"""SQuAD v1.1 EM and F1 metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, List, Sequence

import config

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    """Normalise an answer string for SQuAD v1.1 scoring.

    Lower-case, strip punctuation, remove articles (``a|an|the``) as whole
    words, and collapse runs of whitespace to a single space.
    """

    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TABLE)
    no_articles = _ARTICLES_RE.sub(" ", no_punct)
    return _WHITESPACE_RE.sub(" ", no_articles).strip()


def _f1_one(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _em_one(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _example_em_f1(pred: str, golds: Sequence[str]) -> tuple[float, float]:
    """Per-example EM and F1: the max over gold answers."""

    if not golds:
        empty_match = 1.0 if _normalize(pred) == "" else 0.0
        return empty_match, empty_match
    em = max(_em_one(pred, g) for g in golds)
    f1 = max(_f1_one(pred, g) for g in golds)
    return em, f1


def compute_em_f1(
    predictions: List[str], references: List[List[str]]
) -> Dict[str, float]:
    """Compute mean Exact-Match and F1 over a SQuAD evaluation set.

    Args:
        predictions: One predicted answer string per example, aligned with
            ``references``.
        references: Gold answers per example (SQuAD dev gives several).

    Returns:
        Dict with keys ``em`` and ``f1``, each scaled to ``[0, 100]``.
    """

    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must be aligned"
        )
    if not predictions:
        return {"em": 0.0, "f1": 0.0}

    em_sum = 0.0
    f1_sum = 0.0
    for pred, golds in zip(predictions, references):
        em, f1 = _example_em_f1(pred, golds)
        em_sum += em
        f1_sum += f1
    n = len(predictions)
    return {"em": 100.0 * em_sum / n, "f1": 100.0 * f1_sum / n}


def per_question_type_f1(
    predictions: List[str],
    references: List[List[str]],
    qtypes: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute EM/F1 bucketed by question type.

    Buckets are exactly :data:`config.QUESTION_TYPES`. The ``qtypes`` list
    must come from Stage 1's ``validation_examples["question_type"]``
    column so the bucket definitions stay byte-identical with Stage 1.

    Args:
        predictions: Predicted answer strings.
        references: Gold answers per example.
        qtypes: Pre-tagged question-type label per example (aligned).

    Returns:
        Dict mapping bucket label to ``{"em": float, "f1": float,
        "n": int}``; missing buckets are reported with ``n=0`` and
        ``em=f1=0.0``.
    """

    if not (len(predictions) == len(references) == len(qtypes)):
        raise ValueError(
            "predictions, references, qtypes must all align "
            f"(got {len(predictions)}, {len(references)}, {len(qtypes)})"
        )

    sums: Dict[str, Dict[str, float]] = {
        qt: {"em": 0.0, "f1": 0.0, "n": 0} for qt in config.QUESTION_TYPES
    }
    for pred, golds, qtype in zip(predictions, references, qtypes):
        bucket = qtype if qtype in sums else "other"
        em, f1 = _example_em_f1(pred, golds)
        sums[bucket]["em"] += em
        sums[bucket]["f1"] += f1
        sums[bucket]["n"] += 1

    result: Dict[str, Dict[str, float]] = {}
    for qt, agg in sums.items():
        n = int(agg["n"])
        if n == 0:
            result[qt] = {"em": 0.0, "f1": 0.0, "n": 0}
            continue
        result[qt] = {
            "em": 100.0 * agg["em"] / n,
            "f1": 100.0 * agg["f1"] / n,
            "n": n,
        }
    return result

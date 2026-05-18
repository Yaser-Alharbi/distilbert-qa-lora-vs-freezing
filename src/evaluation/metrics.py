"""SQuAD v1.1 EM and F1 metrics."""

from __future__ import annotations

from typing import Dict, List


def compute_em_f1(predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
    """Compute mean Exact-Match and F1 over a SQuAD evaluation set.

    Follows the official SQuAD v1.1 normalisation (lowercase, strip
    articles and punctuation, normalise whitespace) and takes the max
    over gold answers per question.

    Args:
        predictions: One predicted answer string per question.
        references: Gold answers per question (SQuAD dev gives several).

    Returns:
        Dict with keys ``em`` and ``f1``, each in ``[0, 100]``.
    """

    # TODO(stage 4): implement SQuAD-style EM/F1 with answer normalisation.
    raise NotImplementedError


def per_question_type_f1(
    questions: List[str],
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Compute mean F1 bucketed by leading wh-word.

    Buckets: ``what``, ``who``, ``when``, ``where``, ``why``, ``how``,
    plus ``other`` for everything else.

    Args:
        questions: Question strings.
        predictions: Predicted answer strings.
        references: Gold answers per question.

    Returns:
        Dict mapping bucket label to mean F1 in ``[0, 100]``.
    """

    # TODO(stage 4): bucket by wh-word and reuse compute_em_f1 per bucket.
    raise NotImplementedError

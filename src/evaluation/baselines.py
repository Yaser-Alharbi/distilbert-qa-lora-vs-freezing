"""No-train baselines wired into the same Stage 4 scoring path as the
neural variants.

Each baseline emits ``dict[example_id, str]`` so it flows through the same
``compute_em_f1`` / ``per_question_type_f1`` plumbing as decoded model
spans. ``RandomSpanBaseline`` is stochastic (run across the three seeds and
aggregated with a 95% CI); ``TfidfBaseline`` is deterministic and so is
reported with ``n=1`` and no CI.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

import config
from src.models.baselines import TfidfBaseline as _TfidfModel


class RandomSpanBaseline:
    """Pick a uniformly random contiguous word span per example.

    Spans are at most :data:`config.MAX_ANSWER_LENGTH` whitespace-tokens
    long. Deterministic given *seed* (NumPy default RNG); intended to be
    run once per ``config.SEEDS`` entry so Stage 4 can attach a CI.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def predict_for_examples(
        self, id_to_context: Mapping[str, str]
    ) -> Dict[str, str]:
        """Return ``{example_id: random_span_text}`` over every example."""

        max_len = int(config.MAX_ANSWER_LENGTH)
        predictions: Dict[str, str] = {}
        for example_id, context in id_to_context.items():
            words = context.split()
            n_words = len(words)
            if n_words == 0:
                predictions[example_id] = ""
                continue
            span_len = int(self._rng.integers(1, min(max_len, n_words) + 1))
            start = int(self._rng.integers(0, n_words - span_len + 1))
            predictions[example_id] = " ".join(words[start : start + span_len])
        return predictions


class TfidfBaseline:
    """TF-IDF best-sentence retrieval over the question.

    Thin example-id-keyed wrapper over the Stage 2 helper
    :class:`src.models.baselines.TfidfBaseline` (per the task spec which
    permits calling already-existing helpers).
    """

    def __init__(self) -> None:
        self._inner = _TfidfModel()

    def predict_for_examples(
        self,
        id_to_question: Mapping[str, str],
        id_to_context: Mapping[str, str],
    ) -> Dict[str, str]:
        """Return ``{example_id: best_sentence}`` over every example."""

        predictions: Dict[str, str] = {}
        for example_id, context in id_to_context.items():
            question = id_to_question[example_id]
            predictions[example_id] = self._inner.predict([question], [context])[0]
        return predictions

"""Non-neural baselines for extractive QA on SQuAD v1.1."""

from __future__ import annotations

from typing import Any, List


class RandomSpanBaseline:
    """Predicts a uniformly random contiguous word span from the context."""

    def __init__(self, max_span_len: int = 5, seed: int | None = None) -> None:
        """Initialise the baseline.

        Args:
            max_span_len: Maximum number of words in a sampled span.
            seed: Random seed for deterministic sampling.
        """

        self.max_span_len = max_span_len
        self.seed = seed

    def predict(self, questions: List[str], contexts: List[str]) -> List[str]:
        """Return one predicted span per (question, context) pair.

        Args:
            questions: Question strings (unused; kept for API symmetry).
            contexts: Context paragraphs.

        Returns:
            One predicted answer string per input pair.
        """

        # TODO(stage 2): sample (start, length) per context with self.seed.
        raise NotImplementedError


class TfidfBaseline:
    """Picks the context sentence with highest TF-IDF similarity to the question."""

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        """Initialise the baseline.

        Args:
            ngram_range: n-gram range for ``TfidfVectorizer``.
        """

        self.ngram_range = ngram_range
        self._vectorizer: Any | None = None

    def predict(self, questions: List[str], contexts: List[str]) -> List[str]:
        """Return the best-matching sentence per (question, context) pair.

        Args:
            questions: Question strings.
            contexts: Context paragraphs.

        Returns:
            One predicted answer sentence per input pair.
        """

        # TODO(stage 2): sentence-split contexts, fit TF-IDF, return argmax cosine.
        raise NotImplementedError

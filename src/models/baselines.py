"""Non-neural baselines for extractive QA on SQuAD v1.1.

Both baselines expose the same ``predict(questions, contexts) -> List[str]``
interface so that Stage 4 evaluation can treat them identically to the
post-processed neural model outputs.
"""

from __future__ import annotations

import re
import random
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RandomSpanBaseline:
    """Predicts a uniformly random contiguous word span from the context.

    Interface:
        ``predict(questions, contexts) -> List[str]``
    Deterministic given *seed*. No training required.
    """

    def __init__(self, max_span_len: int = 5, seed: int | None = None) -> None:
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

        rng = random.Random(self.seed)
        predictions: List[str] = []
        for context in contexts:
            words = context.split()
            if not words:
                predictions.append("")
                continue
            span_len = rng.randint(1, min(self.max_span_len, len(words)))
            start = rng.randint(0, len(words) - span_len)
            predictions.append(" ".join(words[start : start + span_len]))
        return predictions


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class TfidfBaseline:
    """Returns the context sentence with highest TF-IDF cosine similarity to the question.

    Sentence splitting uses a simple regex on terminal punctuation followed by
    whitespace. Each (question, context) pair is scored independently — the
    vectoriser is fit per-context on its sentences plus the question.

    Interface:
        ``predict(questions, contexts) -> List[str]``
    Deterministic; no neural training required.
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.ngram_range = ngram_range

    def predict(self, questions: List[str], contexts: List[str]) -> List[str]:
        """Return the best-matching sentence per (question, context) pair.

        Args:
            questions: Question strings.
            contexts: Context paragraphs.

        Returns:
            One predicted answer sentence per input pair.
        """

        predictions: List[str] = []
        for question, context in zip(questions, contexts):
            sentences = _split_sentences(context)
            if not sentences:
                predictions.append("")
                continue
            if len(sentences) == 1:
                predictions.append(sentences[0])
                continue

            vectorizer = TfidfVectorizer(ngram_range=self.ngram_range)
            corpus = sentences + [question]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            question_vec = tfidf_matrix[-1]
            sentence_vecs = tfidf_matrix[:-1]
            scores = cosine_similarity(question_vec, sentence_vecs).flatten()
            best_idx = int(scores.argmax())
            predictions.append(sentences[best_idx])
        return predictions


def _split_sentences(text: str) -> List[str]:
    """Split *text* into sentences on terminal punctuation boundaries."""

    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]

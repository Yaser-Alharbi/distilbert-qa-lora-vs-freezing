"""Evaluation subpackage: EM / F1 metrics, span decoding, baselines, runner."""

from src.evaluation.evaluate import run
from src.evaluation.metrics import compute_em_f1, per_question_type_f1

__all__ = ["compute_em_f1", "per_question_type_f1", "run"]

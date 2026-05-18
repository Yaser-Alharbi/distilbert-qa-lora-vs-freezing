"""Models subpackage: baselines, DistilBERT QA, LoRA adapter, and param utils."""

from src.models.baselines import RandomSpanBaseline, TfidfBaseline
from src.models.distilbert_qa import build_model
from src.models.lora import apply_lora
from src.models.param_utils import count_parameters

__all__ = [
    "RandomSpanBaseline",
    "TfidfBaseline",
    "build_model",
    "apply_lora",
    "count_parameters",
]

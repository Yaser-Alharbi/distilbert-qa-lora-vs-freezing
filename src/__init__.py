"""Top-level package for the ELEC0141 DLNLP Extractive QA project.

Submodules:
    data: SQuAD download, preprocessing and tokenisation utilities.
    models: DistilBERT QA model, LoRA adapter and baseline implementations.
    training: Training loop, optimiser, scheduler and early-stopping logic.
    evaluation: EM / F1 metrics and per-question-type breakdowns.
    plotting: Figure generation for the report.
    utils: Shared helpers (seed control, logging, etc.).
"""

__all__ = [
    "data",
    "models",
    "training",
    "evaluation",
    "plotting",
    "utils",
]

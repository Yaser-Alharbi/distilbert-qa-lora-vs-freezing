"""Training subpackage: per-(variant, seed) training loop with caching."""

from src.training.trainer import (
    log_last_run_summary,
    run_all_training,
    train_variant,
)

__all__ = ["train_variant", "run_all_training", "log_last_run_summary"]

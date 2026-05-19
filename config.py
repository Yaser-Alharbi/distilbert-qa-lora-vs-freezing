"""Central configuration for the ELEC0141 DLNLP Extractive QA project.

This module is the single source of truth for all paths, hyper-parameters,
seeds, model identifiers, and device selection. Every other module imports
from here; nothing else should hard-code values that belong in this file.

Importing this module has one side effect: it ensures that the runtime
output directories (``data/``, ``results/``, ``plots/``) exist on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths (all relative to this file's location, i.e. the repository root)
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
PLOTS_DIR: Path = ROOT_DIR / "plots"
SRC_DIR: Path = ROOT_DIR / "src"

# HuggingFace caches are kept inside ``data/`` so they are gitignored together
# with the rest of the runtime artefacts.
HF_CACHE_DIR: Path = DATA_DIR / "hf_cache"

#: On-disk root for Stage 3 per-(variant, seed) run outputs
#: (predictions.npz, history.json, meta.json).
RUNS_DIR: Path = RESULTS_DIR / "runs"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

#: Seeds used for multi-seed averaging of the main experimental conditions.
SEEDS: List[int] = [42, 1337, 2024]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_NAME: str = "distilbert-base-uncased"

# ---------------------------------------------------------------------------
# Tokenisation / sequence handling
# ---------------------------------------------------------------------------

MAX_SEQ_LEN: int = 384
DOC_STRIDE: int = 128

# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

BATCH_SIZE: int = 16
LR: float = 3e-5
EPOCHS: int = 2
WEIGHT_DECAY: float = 0.01
WARMUP_RATIO: float = 0.1

#: Global gradient-norm clip applied after ``loss.backward()`` and before
#: ``optimizer.step()``. Set to ``0`` (or any non-positive value) to disable.
GRAD_CLIP: float = 1.0

#: How often (in optimizer steps) to log a step-level training/validation
#: loss point to ``history.json`` for the Stage 5 learning-curve plot. An
#: extra point is always logged at the last step of each epoch so every
#: epoch ends with an aligned data point.
EVAL_EVERY_STEPS: int = 200

#: Number of validation features used to compute the step-level
#: ``val_loss``. A fixed subset is sampled with :data:`SEEDS[0]` and reused
#: across every variant and seed so the curves are directly comparable.
#: The per-epoch ``epoch_val_loss`` summary and the final
#: ``predictions.npz`` dump still cover the full validation set.
VAL_LOSS_SUBSET_SIZE: int = 1000

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: Number of SQuAD v1.1 training examples to retain. The full validation
#: split is always used for evaluation.
TRAIN_SUBSET_SIZE: int = 20_000

#: HuggingFace dataset identifier for SQuAD v1.1.
DATASET_NAME: str = "squad"

#: Version tag for the processed-dataset cache.
PROCESSED_DATA_VERSION: str = "v2"

#: On-disk location of the processed (tokenised, tagged) DatasetDict.
PROCESSED_DATA_DIR: Path = DATA_DIR / f"processed_{PROCESSED_DATA_VERSION}"

#: Allowed question-type buckets for tagging and per-type F1 analysis.
#: Order is informative only; ``"other"`` catches anything that does not
#: match a wh-word, ``"which"``, or a yes/no auxiliary.
QUESTION_TYPES: Tuple[str, ...] = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "yes_no",
    "other",
)

#: Leading auxiliaries / copulae that signal a yes/no question.
YES_NO_LEADS: frozenset[str] = frozenset({
    "is", "are", "was", "were",
    "do", "does", "did",
    "can", "could",
    "has", "have", "had",
    "will", "would", "should",
})

# ---------------------------------------------------------------------------
# Experimental conditions
# ---------------------------------------------------------------------------

#: Layer-freezing configurations evaluated for the freezing branch of the
#: ablation. The values are short identifiers used by the model builder to
#: decide which parameters remain trainable.
FREEZE_CONFIGS: Dict[str, str] = {
    "C0": "head_only",
    "C1": "top_1",
    "C2": "top_3",
    "C3": "full",
}

#: LoRA ranks evaluated for the parameter-efficient fine-tuning branch.
#: Chosen so that higher ranks overlap the C1/C2 trainable-param range,
#: giving multiple matched-budget comparison points with layer-freezing.
LORA_RANKS: List[int] = [16, 32, 64, 128, 256]

#: Multiplier applied as ``lora_alpha = LORA_ALPHA_MULTIPLIER * rank`` so that
#: the effective scaling ratio ``alpha / r`` is constant across all ranks.
LORA_ALPHA_MULTIPLIER: int = 2

#: Dropout applied to LoRA adapter layers during training.
LORA_DROPOUT: float = 0.1

#: All DistilBERT linear projections targeted by LoRA adapters (attention +
#: FFN) to maximise the parameter budget reachable at higher ranks.
LORA_TARGET_MODULES: List[str] = [
    "q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2",
]

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Pick the best available accelerator.

    Order of preference: ``mps`` (Apple Silicon) → ``cuda`` → ``cpu``. The
    ``DLNLP_DEVICE`` environment variable can be set to force a specific
    backend (useful for debugging and for the autograder).

    Returns:
        The device string to pass to :class:`torch.device`.
    """

    override = os.environ.get("DLNLP_DEVICE")
    if override:
        return override

    try:
        import torch  # Imported lazily so config import stays cheap.
    except ImportError:
        return "cpu"

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE: str = _detect_device()

# ---------------------------------------------------------------------------
# CI / fast-mode guard
# ---------------------------------------------------------------------------

#: Whether the pipeline should skip long-running stages (the Stage 3
#: training grid in particular) and rely on committed artefacts. Resolved
#: at import time from the environment so the autograder picks it up
#: without any extra wiring; ``main.py`` reassigns it to ``True`` when the
#: ``--fast`` CLI flag is passed.
FAST_MODE: bool = bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("DLNLP_FAST"))

# ---------------------------------------------------------------------------
# Ensure runtime directories exist on import
# ---------------------------------------------------------------------------

for _directory in (DATA_DIR, RESULTS_DIR, PLOTS_DIR, HF_CACHE_DIR, RUNS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


def summary() -> Dict[str, object]:
    """Return a serialisable snapshot of the configuration.

    Useful for logging at the start of a run and for embedding the exact
    configuration used into the saved results JSON.

    Returns:
        A dictionary mapping configuration names to their current values.
    """

    return {
        "root_dir": str(ROOT_DIR),
        "device": DEVICE,
        "fast_mode": FAST_MODE,
        "model_name": MODEL_NAME,
        "seeds": SEEDS,
        "max_seq_len": MAX_SEQ_LEN,
        "doc_stride": DOC_STRIDE,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "grad_clip": GRAD_CLIP,
        "eval_every_steps": EVAL_EVERY_STEPS,
        "val_loss_subset_size": VAL_LOSS_SUBSET_SIZE,
        "train_subset_size": TRAIN_SUBSET_SIZE,
        "freeze_configs": FREEZE_CONFIGS,
        "lora_ranks": LORA_RANKS,
        "lora_alpha_multiplier": LORA_ALPHA_MULTIPLIER,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "runs_dir": str(RUNS_DIR),
    }

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
from typing import Dict, List

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

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: Number of SQuAD v1.1 training examples to retain. The full validation
#: split is always used for evaluation.
TRAIN_SUBSET_SIZE: int = 20_000

#: HuggingFace dataset identifier for SQuAD v1.1.
DATASET_NAME: str = "squad"

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
LORA_RANKS: List[int] = [4, 8, 16]

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
# Ensure runtime directories exist on import
# ---------------------------------------------------------------------------

for _directory in (DATA_DIR, RESULTS_DIR, PLOTS_DIR, HF_CACHE_DIR):
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
        "model_name": MODEL_NAME,
        "seeds": SEEDS,
        "max_seq_len": MAX_SEQ_LEN,
        "doc_stride": DOC_STRIDE,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "train_subset_size": TRAIN_SUBSET_SIZE,
        "freeze_configs": FREEZE_CONFIGS,
        "lora_ranks": LORA_RANKS,
    }

"""Stage 3 training: fine-tune every (variant, seed) combination with caching.

Per (variant, seed) the run directory ``config.RUNS_DIR / "{variant}_seed{seed}"``
holds three files that constitute the contract for Stages 4 and 5:

    predictions.npz
        ``start_logits`` / ``end_logits`` over the full validation feature
        set, the corresponding ``example_ids`` (one per feature), and the
        ``offsets`` + ``context_mask`` needed to decode best spans without
        reloading the model.

    history.json
        Per-epoch summaries (``epoch_train_loss``, ``epoch_val_loss``,
        ``epoch_seconds``) plus step-level diagnostics for the Stage 5
        learning-curve plot:

            ``steps``
                Aligned parallel arrays
                ``{"global_step": [...], "train_loss": [...],
                "val_loss": [...]}`` — one entry per logged point.
                ``train_loss`` is the running mean of the optimizer-step
                losses since the previous logged point (smoothed by
                construction); ``val_loss`` is the mean cross-entropy
                over a *fixed* validation subset of size
                :data:`config.VAL_LOSS_SUBSET_SIZE` sampled with
                ``config.SEEDS[0]`` and reused identically across all
                variants and seeds for direct cross-run comparability.
                Eval is triggered every
                :data:`config.EVAL_EVERY_STEPS` optimizer steps and
                always once more at the last step of each epoch.
            ``epoch_boundaries``
                ``[{"epoch": int, "global_step": int}, ...]`` — the
                cumulative step count at the *end* of each epoch, derived
                from the actual per-epoch step counts so a partial final
                epoch is handled correctly.
            ``meta``
                ``{"total_steps", "steps_per_epoch", "eval_every_steps",
                "epochs"}`` — enough to render the x-axis and overlay the
                epoch-boundary lines without recomputing.

    meta.json
        Variant identifier, seed, parameter counts, training wall-clock
        (training loop only, excluding the final prediction dump), the
        config hash that controls cache invalidation, and a summary of the
        learning-rate schedule + device.

Caching mirrors the Stage 1 processed-data pattern: a SHA-256 hash of every
training-relevant configuration field (model, tokenisation, optimisation,
step-eval cadence, val-loss subset size, variant spec, seed) is written
into ``meta.json``. When a subsequent run finds an existing ``meta.json``
whose hash matches the current configuration, the run is skipped and the
cached artefacts are reused.

Public API
----------
``train_variant(variant_name, seed)``
    Train one combination (or load from cache). Returns the run directory.

``run_all_training()``
    Train every combination of ``config.FREEZE_CONFIGS`` ∪
    ``config.LORA_RANKS`` with every seed in ``config.SEEDS``.

``log_last_run_summary()``
    Emit a formatted summary table for the most recent
    ``run_all_training`` invocation (variant, seed, train/val loss, time,
    cached flag).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

import config
from src.data.loader import load_squad
from src.models.distilbert_qa import build_model
from src.models.lora import apply_lora
from src.models.param_utils import count_parameters
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

_PREDICTIONS_FILE = "predictions.npz"
_HISTORY_FILE = "history.json"
_META_FILE = "meta.json"

# DistilBERT places ``[CLS]`` at position 0 of every encoded sequence; the
# Stage 1 loader follows the same convention when collapsing out-of-window
# answers, so we hard-code the index here to avoid re-loading the tokenizer.
_CLS_INDEX = 0


def _build_variant_registry() -> Dict[str, Tuple[str, Any]]:
    """Map every variant name to a ``(kind, spec)`` constructor descriptor."""

    registry: Dict[str, Tuple[str, Any]] = {}
    for tag in config.FREEZE_CONFIGS:
        registry[tag] = ("freeze", tag)
    for rank in config.LORA_RANKS:
        registry[f"lora_r{rank}"] = ("lora", int(rank))
    return registry


_VARIANT_REGISTRY: Dict[str, Tuple[str, Any]] = _build_variant_registry()

# Process-level caches so that 27 sequential runs only load the processed
# DatasetDict, derive the validation gold positions, and sample the fixed
# val-loss subset once.
_PROCESSED_CACHE: Dict[str, Any] = {}

# Records (variant, seed, cached_before_call) for the most recent
# ``run_all_training`` call so ``log_last_run_summary`` can flag which runs
# were short-circuited by the cache.
_LAST_RUN_STATS: List[Tuple[str, int, bool]] = []


def _load_processed() -> DatasetDict:
    if "data" not in _PROCESSED_CACHE:
        _PROCESSED_CACHE["data"] = load_squad()
    return _PROCESSED_CACHE["data"]


def _val_positions() -> Tuple[List[int], List[int]]:
    if "val_positions" not in _PROCESSED_CACHE:
        data = _load_processed()
        _PROCESSED_CACHE["val_positions"] = _derive_val_positions(
            data["validation"], data["validation_examples"]
        )
    return _PROCESSED_CACHE["val_positions"]


def _val_loss_subset() -> Dataset:
    """Return the deterministic fixed val subset used for step-level loss.

    The subset is sampled with ``config.SEEDS[0]`` over the full
    validation features (with start/end position labels attached), then
    truncated to :data:`config.VAL_LOSS_SUBSET_SIZE`. Identical contents
    across every variant and seed in a single process and across processes
    (HuggingFace ``Dataset.shuffle`` is seeded-deterministic), making the
    step-level ``val_loss`` curves directly comparable across all 27 runs.
    """

    if "val_loss_subset" not in _PROCESSED_CACHE:
        data = _load_processed()
        val_features = data["validation"]
        val_starts, val_ends = _val_positions()
        val_with_positions = val_features.add_column("start_positions", val_starts)
        val_with_positions = val_with_positions.add_column("end_positions", val_ends)
        n = min(int(config.VAL_LOSS_SUBSET_SIZE), len(val_with_positions))
        subset = val_with_positions.shuffle(seed=int(config.SEEDS[0])).select(range(n))
        _PROCESSED_CACHE["val_loss_subset"] = subset
    return _PROCESSED_CACHE["val_loss_subset"]


def _construct_model(variant_name: str) -> torch.nn.Module:
    kind, spec = _VARIANT_REGISTRY[variant_name]
    if kind == "freeze":
        return build_model(spec)
    if kind == "lora":
        base = build_model("C0")
        return apply_lora(base, spec)
    raise ValueError(f"Unknown variant kind {kind!r} for {variant_name!r}")


def _run_dir(variant_name: str, seed: int) -> Path:
    return config.RUNS_DIR / f"{variant_name}_seed{seed}"


def _config_hash(variant_name: str, seed: int) -> str:
    """Return a stable SHA-256 over every config field that affects training."""

    kind, spec = _VARIANT_REGISTRY[variant_name]
    payload: Dict[str, Any] = {
        "model_name": config.MODEL_NAME,
        "max_seq_len": config.MAX_SEQ_LEN,
        "doc_stride": config.DOC_STRIDE,
        "train_subset_size": config.TRAIN_SUBSET_SIZE,
        "processed_data_version": config.PROCESSED_DATA_VERSION,
        "batch_size": config.BATCH_SIZE,
        "epochs": config.EPOCHS,
        "lr": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_ratio": config.WARMUP_RATIO,
        "grad_clip": config.GRAD_CLIP,
        "eval_every_steps": config.EVAL_EVERY_STEPS,
        "val_loss_subset_size": config.VAL_LOSS_SUBSET_SIZE,
        "variant_name": variant_name,
        "variant_kind": kind,
        "variant_spec": spec,
        "seed": seed,
    }
    if kind == "lora":
        payload["lora_alpha_multiplier"] = config.LORA_ALPHA_MULTIPLIER
        payload["lora_dropout"] = config.LORA_DROPOUT
        payload["lora_target_modules"] = list(config.LORA_TARGET_MODULES)
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _is_cached(run_dir: Path, expected_hash: str) -> bool:
    meta_path = run_dir / _META_FILE
    if not meta_path.is_file():
        return False
    if not (run_dir / _PREDICTIONS_FILE).is_file():
        return False
    if not (run_dir / _HISTORY_FILE).is_file():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("config_hash") == expected_hash


def _derive_val_positions(
    val_features: Dataset, val_examples: Dataset
) -> Tuple[List[int], List[int]]:
    """Map each validation feature's gold answer to (start, end) token positions.

    Mirrors the off-window→CLS fallback the Stage 1 loader applies to the
    training features, so per-epoch validation loss is computed on the same
    label semantics. Used only for the diagnostic ``val_loss``; never
    persisted (Stage 4 decodes spans directly from the saved logits).
    """

    id_to_answer = {ex["id"]: ex["answers"] for ex in val_examples}
    offset_mapping = val_features["offset_mapping"]
    example_ids = val_features["example_id"]

    starts: List[int] = []
    ends: List[int] = []
    for offsets, example_id in zip(offset_mapping, example_ids):
        answer = id_to_answer[example_id]
        if not answer["answer_start"]:
            starts.append(_CLS_INDEX)
            ends.append(_CLS_INDEX)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        context_start = 0
        while context_start < len(offsets) and offsets[context_start] is None:
            context_start += 1
        context_end = len(offsets) - 1
        while context_end >= 0 and offsets[context_end] is None:
            context_end -= 1

        if (
            context_start >= len(offsets)
            or context_end < 0
            or offsets[context_start][0] > start_char
            or offsets[context_end][1] < end_char
        ):
            starts.append(_CLS_INDEX)
            ends.append(_CLS_INDEX)
            continue

        idx = context_start
        while idx <= context_end and offsets[idx][0] <= start_char:
            idx += 1
        starts.append(idx - 1)

        idx = context_end
        while idx >= context_start and offsets[idx][1] >= end_char:
            idx -= 1
        ends.append(idx + 1)
    return starts, ends


def _build_dataloaders(
    train_features: Dataset, val_features_with_positions: Dataset, seed: int
) -> Tuple[DataLoader, DataLoader]:
    columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    train_view = train_features.with_format("torch", columns=columns)
    val_view = val_features_with_positions.with_format("torch", columns=columns)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_view,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_view,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, val_loader


def _build_val_loss_loader(subset: Dataset) -> DataLoader:
    """Build a forward-only DataLoader over the fixed val-loss subset."""

    columns = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    view = subset.with_format("torch", columns=columns)
    return DataLoader(
        view,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def _group_parameters(model: torch.nn.Module) -> List[Dict[str, Any]]:
    """Apply weight decay to everything except biases and LayerNorm scales."""

    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": config.WEIGHT_DECAY})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def _build_optimizer_scheduler(
    model: torch.nn.Module, num_training_steps: int
) -> Tuple[torch.optim.Optimizer, Any, int]:
    warmup_steps = int(config.WARMUP_RATIO * num_training_steps)
    optimizer = torch.optim.AdamW(_group_parameters(model), lr=config.LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler, warmup_steps


def _to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {key: tensor.to(device) for key, tensor in batch.items()}


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    starting_global_step: int,
    step_log: Dict[str, List[Any]],
    val_loss_loader: DataLoader,
    eval_every_steps: int,
) -> Tuple[float, int]:
    """Run one training epoch, evaluating on a fixed val subset at intervals.

    Step-level diagnostics are appended to ``step_log`` (parallel arrays
    ``global_step``, ``train_loss``, ``val_loss``). Each entry's
    ``train_loss`` is the running mean over the optimizer steps since the
    previous logged point (smoother than per-batch loss); ``val_loss`` is
    the mean cross-entropy over the fixed val subset. A point is logged
    every ``eval_every_steps`` optimizer steps and again at the last step
    of the epoch, so each epoch always ends with an aligned data point.

    Args:
        starting_global_step: Cumulative optimizer-step count *before* this
            epoch begins (``0`` at the start of training).
        step_log: Dict with keys ``"global_step"``, ``"train_loss"``,
            ``"val_loss"``; one entry is appended per logged point.
        val_loss_loader: DataLoader over the fixed val-loss subset.
        eval_every_steps: Step interval between subset-loss evaluations.

    Returns:
        ``(epoch_train_loss, ending_global_step)`` — the running-mean
        training loss over the *whole* epoch and the cumulative step count
        after the epoch.
    """

    model.train()
    grad_clip = config.GRAD_CLIP if config.GRAD_CLIP and config.GRAD_CLIP > 0 else None
    epoch_loss_sum = 0.0
    epoch_examples = 0
    interval_loss_sum = 0.0
    interval_examples = 0
    global_step = starting_global_step
    n_batches = len(loader)

    progress = tqdm(
        loader,
        desc=f"epoch {epoch}/{total_epochs} train",
        leave=False,
        dynamic_ncols=True,
    )
    for i, batch in enumerate(progress):
        batch = _to_device(batch, device)
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), grad_clip
            )
        optimizer.step()
        scheduler.step()

        global_step += 1
        loss_value = float(loss.item())
        batch_size = batch["input_ids"].size(0)

        epoch_loss_sum += loss_value * batch_size
        epoch_examples += batch_size
        interval_loss_sum += loss_value * batch_size
        interval_examples += batch_size

        is_eval_step = global_step % eval_every_steps == 0
        is_last_step = i == n_batches - 1
        if is_eval_step or is_last_step:
            running_train_loss = interval_loss_sum / max(interval_examples, 1)
            val_loss = _validate(model, val_loss_loader, device)
            model.train()
            step_log["global_step"].append(int(global_step))
            step_log["train_loss"].append(float(running_train_loss))
            step_log["val_loss"].append(float(val_loss))
            interval_loss_sum = 0.0
            interval_examples = 0

        progress.set_postfix(loss=f"{epoch_loss_sum / max(epoch_examples, 1):.4f}")

    return epoch_loss_sum / max(epoch_examples, 1), global_step


@torch.no_grad()
def _validate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0
    seen_examples = 0
    for batch in loader:
        batch = _to_device(batch, device)
        outputs = model(**batch)
        batch_size = batch["input_ids"].size(0)
        total_loss += float(outputs.loss.item()) * batch_size
        seen_examples += batch_size
    return total_loss / max(seen_examples, 1)


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    for batch in loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        outputs = model(**inputs)
        starts.append(outputs.start_logits.detach().to(torch.float32).cpu().numpy())
        ends.append(outputs.end_logits.detach().to(torch.float32).cpu().numpy())
    return np.concatenate(starts, axis=0), np.concatenate(ends, axis=0)


def _pack_offsets(
    offset_mapping: List[List[Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of per-feature offset lists to dense numpy arrays.

    Returns:
        ``(offsets, context_mask)`` where ``offsets`` has shape
        ``(num_features, MAX_SEQ_LEN, 2)`` and ``context_mask`` has shape
        ``(num_features, MAX_SEQ_LEN)``. Positions whose original offset
        was ``None`` (i.e. outside the context segment) get ``(0, 0)`` in
        ``offsets`` and ``False`` in ``context_mask``.
    """

    num_features = len(offset_mapping)
    seq_len = config.MAX_SEQ_LEN
    offsets = np.zeros((num_features, seq_len, 2), dtype=np.int32)
    context_mask = np.zeros((num_features, seq_len), dtype=bool)
    for i, per_feature in enumerate(offset_mapping):
        for k, off in enumerate(per_feature):
            if off is None:
                continue
            offsets[i, k, 0] = int(off[0])
            offsets[i, k, 1] = int(off[1])
            context_mask[i, k] = True
    return offsets, context_mask


def _save_predictions(
    run_dir: Path,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    example_ids: List[str],
    offset_mapping: List[List[Any]],
) -> None:
    offsets, context_mask = _pack_offsets(offset_mapping)
    np.savez_compressed(
        run_dir / _PREDICTIONS_FILE,
        start_logits=start_logits.astype(np.float32),
        end_logits=end_logits.astype(np.float32),
        example_ids=np.array(example_ids, dtype="U"),
        offsets=offsets,
        context_mask=context_mask,
    )


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _release_device_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        empty_cache = getattr(torch.mps, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def _build_history_meta(
    epoch_boundaries: List[Dict[str, int]], configured_epochs: int
) -> Dict[str, Any]:
    """Derive the ``history["meta"]`` block from the epoch-boundary list.

    ``steps_per_epoch`` is reconstructed from the cumulative boundary
    positions, so a partial final epoch (smaller last entry) is handled
    correctly. ``eval_every_steps`` is the per-epoch step count when every
    epoch had the same number of steps, otherwise ``None`` (irregular
    cadence — Stage 5 should fall back to ``epoch_boundaries``).
    """

    if not epoch_boundaries:
        return {
            "total_steps": 0,
            "steps_per_epoch": [],
            "eval_every_steps": None,
            "epochs": int(configured_epochs),
        }

    boundary_steps = [int(b["global_step"]) for b in epoch_boundaries]
    steps_per_epoch = [boundary_steps[0]] + [
        boundary_steps[i] - boundary_steps[i - 1]
        for i in range(1, len(boundary_steps))
    ]
    uniform = all(s == steps_per_epoch[0] for s in steps_per_epoch)
    return {
        "total_steps": boundary_steps[-1],
        "steps_per_epoch": steps_per_epoch,
        "eval_every_steps": steps_per_epoch[0] if uniform else None,
        "epochs": int(configured_epochs),
    }


def train_variant(variant_name: str, seed: int) -> Path:
    """Train one (variant, seed) combination or short-circuit on cache hit.

    Args:
        variant_name: Key from the freezing+LoRA registry, e.g. ``"C2"``
            or ``"lora_r64"``.
        seed: Global seed used for Python/NumPy/PyTorch (incl. DataLoader
            shuffle) and applied *before* model construction so that the
            randomly initialised QA head and LoRA matrices are reproducible.

    Returns:
        The run directory containing ``predictions.npz``, ``history.json``
        and ``meta.json``.
    """

    if variant_name not in _VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown variant {variant_name!r}; "
            f"expected one of {sorted(_VARIANT_REGISTRY)}"
        )

    run_dir = _run_dir(variant_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = _config_hash(variant_name, seed)

    if _is_cached(run_dir, cfg_hash):
        logger.info(
            "  %-12s seed=%-4d  cached, skipping (dir=%s)",
            variant_name, seed, run_dir,
        )
        return run_dir

    processed = _load_processed()
    train_features = processed["train"]
    val_features = processed["validation"]
    val_starts, val_ends = _val_positions()
    val_with_positions = val_features.add_column("start_positions", val_starts)
    val_with_positions = val_with_positions.add_column("end_positions", val_ends)

    set_seed(seed)
    model = _construct_model(variant_name)
    trainable, total = count_parameters(model)

    device = torch.device(config.DEVICE)
    model.to(device)

    train_loader, val_loader = _build_dataloaders(
        train_features, val_with_positions, seed
    )
    val_loss_loader = _build_val_loss_loader(_val_loss_subset())
    num_training_steps = config.EPOCHS * len(train_loader)
    optimizer, scheduler, warmup_steps = _build_optimizer_scheduler(
        model, num_training_steps
    )

    logger.info(
        "  %-12s seed=%-4d  training start: trainable=%d/%d  steps=%d  "
        "warmup=%d  eval_every=%d  val_subset=%d  device=%s",
        variant_name, seed, trainable, total, num_training_steps, warmup_steps,
        int(config.EVAL_EVERY_STEPS), len(val_loss_loader.dataset), device,
    )

    history: Dict[str, Any] = {
        "epoch_train_loss": [],
        "epoch_val_loss": [],
        "epoch_seconds": [],
        "steps": {"global_step": [], "train_loss": [], "val_loss": []},
        "epoch_boundaries": [],
    }

    global_step = 0
    t_train_start = time.perf_counter()
    for epoch in range(1, config.EPOCHS + 1):
        epoch_t0 = time.perf_counter()
        train_loss, global_step = _train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, config.EPOCHS, global_step, history["steps"],
            val_loss_loader, int(config.EVAL_EVERY_STEPS),
        )
        val_loss = _validate(model, val_loader, device)
        epoch_dt = time.perf_counter() - epoch_t0
        history["epoch_train_loss"].append(train_loss)
        history["epoch_val_loss"].append(val_loss)
        history["epoch_seconds"].append(epoch_dt)
        history["epoch_boundaries"].append(
            {"epoch": int(epoch), "global_step": int(global_step)}
        )
        logger.info(
            "  %-12s seed=%-4d  epoch %d/%d  train_loss=%.4f  val_loss=%.4f  (%.1fs)",
            variant_name, seed, epoch, config.EPOCHS, train_loss, val_loss, epoch_dt,
        )
    train_wall_clock_sec = time.perf_counter() - t_train_start

    history["meta"] = _build_history_meta(history["epoch_boundaries"], config.EPOCHS)

    start_logits, end_logits = _collect_logits(model, val_loader, device)
    _save_predictions(
        run_dir,
        start_logits=start_logits,
        end_logits=end_logits,
        example_ids=list(val_features["example_id"]),
        offset_mapping=list(val_features["offset_mapping"]),
    )
    _save_json(run_dir / _HISTORY_FILE, history)

    kind, spec = _VARIANT_REGISTRY[variant_name]
    meta = {
        "variant": variant_name,
        "variant_kind": kind,
        "variant_spec": spec,
        "seed": int(seed),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "train_wall_clock_sec": float(train_wall_clock_sec),
        "num_train_features": int(len(train_features)),
        "num_val_features": int(len(val_features)),
        "num_val_loss_subset": int(len(val_loss_loader.dataset)),
        "num_training_steps": int(num_training_steps),
        "epochs": int(config.EPOCHS),
        "batch_size": int(config.BATCH_SIZE),
        "lr": float(config.LR),
        "weight_decay": float(config.WEIGHT_DECAY),
        "grad_clip": float(config.GRAD_CLIP),
        "eval_every_steps": int(config.EVAL_EVERY_STEPS),
        "val_loss_subset_size": int(config.VAL_LOSS_SUBSET_SIZE),
        "lr_schedule": {
            "type": "linear_warmup_then_linear_decay",
            "warmup_steps": int(warmup_steps),
            "warmup_ratio": float(config.WARMUP_RATIO),
            "total_steps": int(num_training_steps),
        },
        "device": str(device),
        "model_name": config.MODEL_NAME,
        "max_seq_len": int(config.MAX_SEQ_LEN),
        "doc_stride": int(config.DOC_STRIDE),
        "train_subset_size": int(config.TRAIN_SUBSET_SIZE),
        "processed_data_version": config.PROCESSED_DATA_VERSION,
        "config_hash": cfg_hash,
    }
    _save_json(run_dir / _META_FILE, meta)

    del model, optimizer, scheduler, train_loader, val_loader, val_loss_loader
    _release_device_memory(device)

    return run_dir


def run_all_training() -> List[Path]:
    """Train every (variant, seed) combination with caching.

    Iterates the variant registry (4 freezing + 5 LoRA = 9 variants) across
    :data:`config.SEEDS` (3 seeds) for a total of 27 runs and dispatches to
    :func:`train_variant`. Each run's cache status is recorded so that
    :func:`log_last_run_summary` can report which combinations were
    short-circuited.

    Returns:
        One :class:`Path` per (variant, seed) — fresh or cached.
    """

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _LAST_RUN_STATS.clear()

    variant_names = list(_VARIANT_REGISTRY)

    if config.FAST_MODE:
        logger.info("fast mode: skipping stage 3 training grid")
        missing = [
            artefact
            for variant in variant_names
            for seed in config.SEEDS
            for artefact in (
                _run_dir(variant, seed) / _META_FILE,
                _run_dir(variant, seed) / _HISTORY_FILE,
            )
            if not artefact.is_file()
        ]
        if missing:
            logger.warning(
                "fast mode: %d expected committed artefact(s) absent; "
                "downstream stages may degrade",
                len(missing),
            )
        return []

    total_runs = len(variant_names) * len(config.SEEDS)
    logger.info(
        "Stage 3 schedule: %d variants x %d seeds = %d total runs",
        len(variant_names), len(config.SEEDS), total_runs,
    )

    run_dirs: List[Path] = []
    for variant_name in variant_names:
        for seed in config.SEEDS:
            run_dir = _run_dir(variant_name, seed)
            cfg_hash = _config_hash(variant_name, seed)
            was_cached = _is_cached(run_dir, cfg_hash)
            train_variant(variant_name, seed)
            _LAST_RUN_STATS.append((variant_name, int(seed), was_cached))
            run_dirs.append(run_dir)

    return run_dirs


def log_last_run_summary() -> None:
    """Log a formatted table for the most recent ``run_all_training`` call.

    Columns: ``variant``, ``seed``, ``train_loss`` (final epoch),
    ``val_loss`` (final epoch), ``time_s`` (training wall-clock excluding
    the prediction dump), and ``cached`` (yes/no).
    """

    if not _LAST_RUN_STATS:
        logger.info("No training runs recorded in this session.")
        return

    logger.info("--- Stage 3 summary ---")
    header = "  %-12s %5s  %12s  %12s  %10s  %8s"
    row = "  %-12s %5d  %12.4f  %12.4f  %10.1f  %8s"
    logger.info(header, "variant", "seed", "train_loss", "val_loss", "time_s", "cached")
    for variant_name, seed, was_cached in _LAST_RUN_STATS:
        run_dir = _run_dir(variant_name, seed)
        with (run_dir / _META_FILE).open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        with (run_dir / _HISTORY_FILE).open("r", encoding="utf-8") as fh:
            history = json.load(fh)
        logger.info(
            row,
            variant_name,
            seed,
            history["epoch_train_loss"][-1],
            history["epoch_val_loss"][-1],
            meta["train_wall_clock_sec"],
            "yes" if was_cached else "no",
        )

"""SQuAD v1.1 dataset loading and preprocessing.

Downloads SQuAD v1.1 via HuggingFace ``datasets``, deterministically subsets
the training split, tokenises question/context pairs with a sliding window,
maps gold answer character spans to token start/end positions, tags each
original example with a wh-word question-type bucket, and persists the
processed :class:`datasets.DatasetDict` together with a
``dataset_stats.json`` summary.

The processed cache lives under :data:`config.PROCESSED_DATA_DIR` and is
keyed by :data:`config.PROCESSED_DATA_VERSION`; rerunning
:func:`load_squad` when that cache exists is a pure disk read.

Splits returned:

    ``train``                tokenised features with
                             ``start_positions`` / ``end_positions`` /
                             ``is_impossible``.
    ``validation``           tokenised features with ``example_id`` and
                             ``offset_mapping`` (set to ``None`` outside
                             the context segment so eval can ignore those
                             positions).
    ``train_examples``       raw subset rows + ``question_type``.
    ``validation_examples``  full raw validation rows + ``question_type``.
"""

from __future__ import annotations

import json
import logging
import string
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import config

logger = logging.getLogger(__name__)

_STATS_FILENAME = "dataset_stats.json"
_WH_WORDS = frozenset(
    qt for qt in config.QUESTION_TYPES if qt not in ("other", "yes_no")
)


def load_squad() -> DatasetDict:
    """Load, subset, tokenise, and tag SQuAD v1.1.

    Reads from :data:`config.PROCESSED_DATA_DIR` when a cache exists.
    Otherwise downloads SQuAD v1.1 through HuggingFace ``datasets``,
    seeded-subsets the training split to
    :data:`config.TRAIN_SUBSET_SIZE`, asserts no train/validation id
    leakage, runs sliding-window DistilBERT tokenisation, maps gold
    answer character spans to token indices (collapsing answers that
    fall outside a feature window to the CLS index), tags each original
    example with a wh-word question type, persists the resulting
    :class:`datasets.DatasetDict`, and writes a stats summary to
    :data:`config.RESULTS_DIR`.

    Returns:
        A :class:`datasets.DatasetDict` with four splits described in the
        module docstring.
    """

    cache_dir = config.PROCESSED_DATA_DIR
    if _cache_exists(cache_dir):
        logger.info("Loading processed SQuAD cache from %s", cache_dir)
        return load_from_disk(str(cache_dir))

    logger.info("No processed cache at %s; rebuilding from raw SQuAD", cache_dir)
    raw = load_dataset(config.DATASET_NAME, cache_dir=str(config.HF_CACHE_DIR))
    raw_train_size = len(raw["train"])
    raw_validation_size = len(raw["validation"])

    seed = config.SEEDS[0]
    target_subset = min(config.TRAIN_SUBSET_SIZE, raw_train_size)
    train_subset = raw["train"].shuffle(seed=seed).select(range(target_subset))
    validation = raw["validation"]
    logger.info(
        "Splits: raw_train=%d -> subset=%d (seed=%d); validation=%d (untouched)",
        raw_train_size,
        len(train_subset),
        seed,
        len(validation),
    )

    _assert_no_id_leakage(train_subset, validation)

    train_examples = _tag_question_types(train_subset)
    validation_examples = _tag_question_types(validation)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise RuntimeError(
            "DistilBERT fast tokenizer required for offset mapping; "
            f"got {type(tokenizer).__name__}"
        )

    train_features = train_examples.map(
        _prepare_train_features,
        batched=True,
        remove_columns=train_examples.column_names,
        fn_kwargs={"tokenizer": tokenizer},
        desc="tokenise/train",
    )
    validation_features = validation_examples.map(
        _prepare_validation_features,
        batched=True,
        remove_columns=validation_examples.column_names,
        fn_kwargs={"tokenizer": tokenizer},
        desc="tokenise/validation",
    )

    impossible_count = int(sum(train_features["is_impossible"]))
    impossible_pct = (
        100.0 * impossible_count / len(train_features) if len(train_features) else 0.0
    )
    logger.info(
        "Train features: %d (impossible spans after windowing: %d, %.2f%%)",
        len(train_features),
        impossible_count,
        impossible_pct,
    )
    logger.info("Validation features: %d", len(validation_features))

    processed = DatasetDict(
        {
            "train": train_features,
            "validation": validation_features,
            "train_examples": train_examples,
            "validation_examples": validation_examples,
        }
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    processed.save_to_disk(str(cache_dir))
    logger.info("Saved processed DatasetDict to %s", cache_dir)

    stats = _build_stats(
        raw_train_size=raw_train_size,
        raw_validation_size=raw_validation_size,
        train_examples=train_examples,
        validation_examples=validation_examples,
        train_features=train_features,
        validation_features=validation_features,
        impossible_count=impossible_count,
        impossible_pct=impossible_pct,
        seed=seed,
    )
    _write_stats(stats)

    return processed


@lru_cache(maxsize=1)
def validation_example_index() -> Tuple[
    List[str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, List[str]],
    Dict[str, str],
]:
    """Return aligned example-level lookups for the validation split.

    Pure cache read on top of :func:`load_squad`'s
    ``validation_examples`` split — no re-download, no re-tokenisation.
    The first element pins the canonical iteration order; the four
    mappings share that key set.

    Returns:
        ``(ordered_ids, id_to_context, id_to_question, id_to_golds,
        id_to_qtype)``.

    Notes:
        The result is memoised for the lifetime of the process; callers
        must not mutate the returned containers.
    """

    processed = load_squad()
    val_examples = processed["validation_examples"]
    ordered_ids: List[str] = list(val_examples["id"])
    contexts: List[str] = list(val_examples["context"])
    questions: List[str] = list(val_examples["question"])
    answers: List[Dict[str, Any]] = list(val_examples["answers"])
    qtypes: List[str] = list(val_examples["question_type"])

    id_to_context = dict(zip(ordered_ids, contexts))
    id_to_question = dict(zip(ordered_ids, questions))
    id_to_golds = {eid: list(ans["text"]) for eid, ans in zip(ordered_ids, answers)}
    id_to_qtype = dict(zip(ordered_ids, qtypes))
    return ordered_ids, id_to_context, id_to_question, id_to_golds, id_to_qtype


def _cache_exists(path: Path) -> bool:
    return path.is_dir() and (path / "dataset_dict.json").is_file()


def _assert_no_id_leakage(train: Dataset, validation: Dataset) -> None:
    overlap = set(train["id"]) & set(validation["id"])
    if overlap:
        sample = list(overlap)[:5]
        raise RuntimeError(
            f"Train/validation id leakage: {len(overlap)} shared id(s); "
            f"first few: {sample}"
        )


def _classify_question(question: str) -> str:
    stripped = question.strip()
    if not stripped:
        return "other"
    leading = stripped.split(None, 1)[0].lower().strip(string.punctuation)
    if leading in _WH_WORDS:
        return leading
    if leading in config.YES_NO_LEADS:
        return "yes_no"
    return "other"


def _tag_question_types(examples: Dataset) -> Dataset:
    tags = [_classify_question(q) for q in examples["question"]]
    return examples.add_column("question_type", tags)


def _prepare_train_features(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerFast,
) -> Dict[str, List[Any]]:
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=config.MAX_SEQ_LEN,
        stride=config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions: List[int] = []
    end_positions: List[int] = []
    is_impossible: List[bool] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]

        if len(answer["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            is_impossible.append(True)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        answer_in_window = (
            context_start < len(sequence_ids)
            and context_end >= 0
            and offsets[context_start][0] <= start_char
            and offsets[context_end][1] >= end_char
        )
        if not answer_in_window:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            is_impossible.append(True)
            continue

        idx = context_start
        while idx <= context_end and offsets[idx][0] <= start_char:
            idx += 1
        start_positions.append(idx - 1)

        idx = context_end
        while idx >= context_start and offsets[idx][1] >= end_char:
            idx -= 1
        end_positions.append(idx + 1)
        is_impossible.append(False)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized["is_impossible"] = is_impossible
    return tokenized


def _prepare_validation_features(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizerFast,
) -> Dict[str, List[Any]]:
    questions = [q.lstrip() for q in examples["question"]]
    tokenized = tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=config.MAX_SEQ_LEN,
        stride=config.DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    example_ids: List[str] = []
    masked_offsets: List[List[Optional[List[int]]]] = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        example_ids.append(examples["id"][sample_mapping[i]])
        masked_offsets.append(
            [
                offset if sequence_ids[k] == 1 else None
                for k, offset in enumerate(tokenized["offset_mapping"][i])
            ]
        )

    tokenized["offset_mapping"] = masked_offsets
    tokenized["example_id"] = example_ids
    return tokenized


def _question_type_distribution(examples: Dataset) -> Dict[str, int]:
    counts = Counter(examples["question_type"])
    return {qt: int(counts.get(qt, 0)) for qt in config.QUESTION_TYPES}


def _build_stats(
    *,
    raw_train_size: int,
    raw_validation_size: int,
    train_examples: Dataset,
    validation_examples: Dataset,
    train_features: Dataset,
    validation_features: Dataset,
    impossible_count: int,
    impossible_pct: float,
    seed: int,
) -> Dict[str, Any]:
    return {
        "dataset_name": config.DATASET_NAME,
        "source": "huggingface.datasets.load_dataset",
        "tokenizer_name": config.MODEL_NAME,
        "max_seq_len": config.MAX_SEQ_LEN,
        "doc_stride": config.DOC_STRIDE,
        "seed": seed,
        "cache_version": config.PROCESSED_DATA_VERSION,
        "raw_train_size": raw_train_size,
        "raw_validation_size": raw_validation_size,
        "train_subset_size": len(train_examples),
        "validation_size": len(validation_examples),
        "train_features": len(train_features),
        "validation_features": len(validation_features),
        "impossible_span_count": impossible_count,
        "impossible_span_pct": round(impossible_pct, 4),
        "question_type_distribution": {
            "train": _question_type_distribution(train_examples),
            "validation": _question_type_distribution(validation_examples),
        },
    }


def _write_stats(stats: Dict[str, Any]) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / _STATS_FILENAME
    with out.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
    logger.info("Wrote dataset stats to %s", out)

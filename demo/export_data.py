"""Export ``demo/data.json`` for the static demo page.

Reads the committed Stage 4 metrics (``results/metrics.json``), the per-run
``meta.json`` files and ``results/dataset_stats.json``, then decodes the
local seed-42 ``predictions.npz`` of every variant for a handful of curated
validation questions. Nothing under ``results/`` is written, and
``main.py`` never imports this module.

Usage (from the repository root)::

    python -m demo.export_data                     # write demo/data.json
    python -m demo.export_data --list-candidates   # print example candidates
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import config

logger = logging.getLogger(__name__)

DEMO_DIR: Path = Path(__file__).resolve().parent
OUTPUT_JSON: Path = DEMO_DIR / "data.json"
DATASET_STATS_JSON: Path = config.RESULTS_DIR / "dataset_stats.json"

#: Seed whose predictions supply the example answers.
EXAMPLE_SEED: int = config.SEEDS[0]

#: Curated validation questions, in display order, with an optional
#: hand-written note. Picked from ``--list-candidates`` output: one case where
#: every trained variant above ``C0`` agrees, three where small budgets give
#: different answers, and one where ``lora_r16`` is wrong.
EXAMPLES: Sequence[Mapping[str, str]] = (
    {"id": "5725b56589a1e219009abd23", "note": ""},
    {"id": "56beb86b3aeaaa14008c92bf", "note": ""},
    {"id": "5726a4a9708984140094ccb7", "note": ""},
    {"id": "5730aef38ab72b1400f9c685", "note": ""},
    {
        "id": "56f7f15aa6d7ea1400e172ef",
        "note": (
            "The passage names two degrees a year apart in one sentence; "
            "rank-16 LoRA, like top-1 and top-3 freezing, returns the 1508 "
            "degree instead of the 1509 one."
        ),
    },
)

#: Only short passages are offered as candidates so the page can show them whole.
_MAX_CANDIDATE_CONTEXT_CHARS = 650
_CANDIDATES_PER_GROUP = 5


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _kind(variant: str) -> str:
    return "freeze" if variant in config.FREEZE_CONFIGS else "lora"


def _variant_rows(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name in config.VARIANTS:
        block = metrics["variants"][name]
        meta = _read_json(config.RUNS_DIR / f"{name}_seed{EXAMPLE_SEED}" / "meta.json")
        rows.append(
            {
                "name": name,
                "kind": _kind(name),
                "spec": meta["variant_spec"]
                if _kind(name) == "lora"
                else config.FREEZE_CONFIGS[name],
                "trainable_params": block["trainable_params"],
                "f1": {
                    key: block["f1"][key] for key in ("mean", "ci_low", "ci_high", "n")
                },
                "train_wall_clock_sec": block["train_wall_clock_sec"]["mean"],
                "lr": meta["lr"],
                "device": meta["device"],
            }
        )
    return rows


def _baseline_rows(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": name, "f1": block["f1"]["mean"]}
        for name, block in metrics["baselines"].items()
    ]


def _decode_all() -> Dict[str, Any]:
    """Decode every variant's example-seed predictions over the dev set."""

    if not config.PROCESSED_DATA_DIR.is_dir():
        # load_squad() would otherwise rebuild the cache and rewrite
        # results/dataset_stats.json; this script must not touch results/.
        raise FileNotFoundError(
            f"processed data cache missing: {config.PROCESSED_DATA_DIR}; "
            "run `python main.py` once first"
        )

    # Imported lazily: both pull in datasets/transformers.
    from src.data.loader import (  # pylint: disable=import-outside-toplevel
        validation_example_index,
    )
    from src.evaluation.baselines import (  # pylint: disable=import-outside-toplevel
        RandomSpanBaseline,
        TfidfBaseline,
    )
    from src.evaluation.postprocess import (  # pylint: disable=import-outside-toplevel
        decode_predictions,
    )

    ordered_ids, contexts, questions, golds, qtypes = validation_example_index()
    predictions = {}
    for name in config.VARIANTS:
        npz_path = config.RUNS_DIR / f"{name}_seed{EXAMPLE_SEED}" / "predictions.npz"
        predictions[name], _ = decode_predictions(npz_path, contexts)
        logger.info("decoded %s", npz_path)
    predictions["random_span"] = RandomSpanBaseline(
        seed=EXAMPLE_SEED
    ).predict_for_examples(contexts)
    predictions["tfidf"] = TfidfBaseline().predict_for_examples(questions, contexts)
    return {
        "ids": ordered_ids,
        "contexts": contexts,
        "questions": questions,
        "golds": golds,
        "qtypes": qtypes,
        "predictions": predictions,
    }


def _example_f1(prediction: str, golds: List[str]) -> float:
    from src.evaluation.metrics import (  # pylint: disable=import-outside-toplevel
        compute_em_f1,
    )

    return compute_em_f1([prediction], [golds])["f1"]


def _example_rows(decoded: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for spec in EXAMPLES:
        example_id = spec["id"]
        context = decoded["contexts"][example_id]
        golds = decoded["golds"][example_id]
        answers = {}
        for name in decoded["predictions"]:
            text = decoded["predictions"][name].get(example_id, "")
            answers[name] = {
                "text": text,
                "start": context.find(text) if text else -1,
                "correct": _example_f1(text, golds) == 100.0,
            }
        rows.append(
            {
                "id": example_id,
                "question": decoded["questions"][example_id],
                "question_type": decoded["qtypes"][example_id],
                "context": context,
                "golds": golds,
                "gold_start": context.find(golds[0]),
                "answers": answers,
                "note": spec["note"],
            }
        )
    return rows


def list_candidates() -> None:
    """Log candidate example ids for each curation group."""

    decoded = _decode_all()
    ids = [
        i for i in decoded["ids"]
        if len(decoded["contexts"][i]) < _MAX_CANDIDATE_CONTEXT_CHARS
    ]

    def correct(name: str, example_id: str) -> bool:
        text = decoded["predictions"][name].get(example_id, "")
        return _example_f1(text, decoded["golds"][example_id]) == 100.0

    trained = [v for v in config.VARIANTS if v != "C0"]
    lora = [v for v in config.VARIANTS if _kind(v) == "lora"]
    groups = {
        "all_agree": lambda i: all(correct(v, i) for v in trained),
        "budgets_differ": lambda i: not correct("C1", i)
        and all(correct(v, i) for v in lora + ["C3"]),
        "lora_r16_wrong": lambda i: not correct("lora_r16", i)
        and correct("lora_r64", i),
    }
    for group, rule in groups.items():
        hits = [i for i in ids if rule(i)]
        logger.info("%s: %d candidates", group, len(hits))
        for example_id in hits[:_CANDIDATES_PER_GROUP]:
            logger.info("  %s  %s", example_id, decoded["questions"][example_id])


def export(path: Path = OUTPUT_JSON) -> Path:
    """Write the demo data file and return its path."""

    metrics = _read_json(config.METRICS_JSON)
    stats = _read_json(DATASET_STATS_JSON)
    payload = {
        "model_name": config.MODEL_NAME,
        "dataset": {
            "name": "SQuAD",
            "full_name": "Stanford Question Answering Dataset",
            "train_size": stats["train_subset_size"],
            "validation_size": stats["validation_size"],
        },
        "seeds": metrics["config"]["seeds"],
        "ci_confidence": metrics["config"]["ci_confidence"],
        "example_seed": EXAMPLE_SEED,
        "variants": _variant_rows(metrics),
        "baselines": _baseline_rows(metrics),
        "matched_budget": [
            {
                "lora": pair["a"],
                "freeze": pair["b"],
                "delta_f1": pair["delta_f1"],
            }
            for pair in metrics["matched_budget"]
        ],
        "examples": _example_rows(_decode_all()),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    logger.info("wrote %s", path)
    return path


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="print candidate example ids instead of writing data.json",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.list_candidates:
        list_candidates()
    else:
        export()
    return 0


if __name__ == "__main__":
    sys.exit(main())

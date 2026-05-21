"""Stage 4 orchestration: aggregate Stage 3 logits into a metrics report.

Three operating modes are resolved at entry, all routed through the same
scoring path:

* ``compute`` — every ``predictions.npz`` is on disk. Decode, score, and
  write ``metrics.json`` + ``metrics.csv``. A manifest hash over the
  consumed predictions and the relevant config constants is embedded so
  subsequent runs short-circuit to a cache hit when nothing has changed.
* ``fast`` — at least one ``predictions.npz`` is missing but
  ``metrics.json`` is committed. Load and return it. This is the grader
  path after a fresh checkout (the large ``predictions.npz`` artefacts
  are gitignored) and must exit in <2 s.
* ``force`` — ``EVAL_FORCE_RECOMPUTE`` (or ``--force-eval``) is set. Ignore
  any committed ``metrics.json`` and recompute; raise a clear error if any
  predictions file is missing.

Every neural quantity is reported with a 95% Student-t confidence interval
across :data:`config.SEEDS`. The two no-train baselines
(:class:`RandomSpanBaseline`, :class:`TfidfBaseline`) flow through the
identical scoring path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from scipy import stats

import config
from src.data.loader import validation_example_index
from src.evaluation.baselines import RandomSpanBaseline, TfidfBaseline
from src.evaluation.metrics import compute_em_f1, per_question_type_f1
from src.evaluation.postprocess import decode_predictions

logger = logging.getLogger(__name__)

_PREDICTIONS_FILE = "predictions.npz"
_META_FILE = "meta.json"

_LATENCY_NOTE = (
    "Model forward latency is variant-invariant by construction: LoRA "
    "adapters are mergeable into the same DistilBERT linear projections "
    "and freezing is architecture-invariant. The two Stage-4-measurable "
    "cost signals that discriminate variants are span-decoding latency "
    "and training wall-clock."
)


def _variants() -> List[str]:
    """All experimental variant names in a fixed reportable order."""

    return list(config.FREEZE_CONFIGS) + [f"lora_r{r}" for r in config.LORA_RANKS]


def _run_dir(variant: str, seed: int) -> Path:
    return config.RUNS_DIR / f"{variant}_seed{seed}"


def _collect_run_paths() -> Dict[Tuple[str, int], Dict[str, Path]]:
    """Map ``(variant, seed)`` to predictions/meta paths (no existence check)."""

    paths: Dict[Tuple[str, int], Dict[str, Path]] = {}
    for variant in _variants():
        for seed in config.SEEDS:
            run_dir = _run_dir(variant, int(seed))
            paths[(variant, int(seed))] = {
                "predictions": run_dir / _PREDICTIONS_FILE,
                "meta": run_dir / _META_FILE,
            }
    return paths


def _missing_predictions(
    run_paths: Mapping[Tuple[str, int], Mapping[str, Path]],
) -> List[Path]:
    return [p["predictions"] for p in run_paths.values() if not p["predictions"].is_file()]


def _resolve_mode(
    run_paths: Mapping[Tuple[str, int], Mapping[str, Path]],
    *,
    force: bool,
    metrics_json: Path,
) -> str:
    """Decide between ``compute``, ``fast``, and ``force`` resolution.

    Raises:
        FileNotFoundError: When ``force`` is set but a prediction file is
            missing, or when neither predictions nor a committed metrics
            file are available (Stage 4 has nothing to score against).
    """

    missing = _missing_predictions(run_paths)
    if force:
        if missing:
            raise FileNotFoundError(
                "--force-eval requested but predictions.npz is missing for "
                f"{len(missing)} run(s); first missing: {missing[0]}"
            )
        return "compute"
    if missing:
        if metrics_json.is_file():
            return "fast"
        raise FileNotFoundError(
            f"Stage 4 has no predictions and no committed {metrics_json.name}: "
            f"{len(missing)} prediction file(s) absent; first missing: "
            f"{missing[0]}"
        )
    return "compute"


def _compute_manifest(
    run_paths: Mapping[Tuple[str, int], Mapping[str, Path]],
) -> str:
    """SHA-256 hash over consumed predictions + relevant config constants.

    Uses ``(size, mtime_ns)`` per ``predictions.npz`` rather than file
    content so the manifest is sub-second to compute even with ~750 MB of
    predictions across 27 runs (per the spec's explicit allowance).
    """

    files_payload = {
        f"{variant}_seed{seed}": [
            int(paths["predictions"].stat().st_size),
            int(paths["predictions"].stat().st_mtime_ns),
        ]
        for (variant, seed), paths in sorted(run_paths.items())
    }
    payload = {
        "files": files_payload,
        "n_best_size": int(config.N_BEST_SIZE),
        "max_answer_length": int(config.MAX_ANSWER_LENGTH),
        "ci_confidence": float(config.CI_CONFIDENCE),
        "seeds": list(config.SEEDS),
        "variants": _variants(),
        "metrics_code_version": config.METRICS_CODE_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _aligned_lists(
    predictions: Mapping[str, str],
    ordered_ids: Sequence[str],
    id_to_golds: Mapping[str, List[str]],
    id_to_qtype: Mapping[str, str],
) -> Tuple[List[str], List[List[str]], List[str]]:
    preds: List[str] = []
    refs: List[List[str]] = []
    qtypes: List[str] = []
    for example_id in ordered_ids:
        preds.append(predictions.get(example_id, ""))
        refs.append(id_to_golds[example_id])
        qtypes.append(id_to_qtype[example_id])
    return preds, refs, qtypes


def _aggregate_ci(values: Sequence[float]) -> Dict[str, Any]:
    """Mean/std/95% Student-t CI for a multi-seed sequence.

    ``n == 1`` -> CI fields are ``None`` (no spread to estimate). ``n >= 2``
    -> ``t.ppf((1+confidence)/2, df=n-1)`` × ``std / sqrt(n)``.
    """

    clean = [float(v) for v in values]
    n = len(clean)
    if n == 0:
        return {
            "mean": None, "std": None, "n": 0,
            "ci_low": None, "ci_high": None, "ci_half_width": None,
        }
    mean = sum(clean) / n
    if n == 1:
        return {
            "mean": mean, "std": 0.0, "n": 1,
            "ci_low": None, "ci_high": None, "ci_half_width": None,
        }
    variance = sum((v - mean) ** 2 for v in clean) / (n - 1)
    std = math.sqrt(variance)
    t_crit = float(stats.t.ppf((1.0 + config.CI_CONFIDENCE) / 2.0, df=n - 1))
    half_width = t_crit * std / math.sqrt(n)
    return {
        "mean": mean, "std": std, "n": n,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "ci_half_width": half_width,
    }


def _read_meta(meta_path: Path) -> Dict[str, Any]:
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _score_variant(
    variant: str,
    seed_paths: Mapping[int, Mapping[str, Path]],
    ordered_ids: Sequence[str],
    id_to_context: Mapping[str, str],
    id_to_golds: Mapping[str, List[str]],
    id_to_qtype: Mapping[str, str],
) -> Dict[str, Any]:
    """Score one variant across all of its seeds and aggregate with CIs."""

    per_seed: Dict[str, Dict[str, float]] = {}
    em_values: List[float] = []
    f1_values: List[float] = []
    decode_ms_values: List[float] = []
    train_wall_clock_values: List[float] = []
    qtype_em: Dict[str, List[float]] = {qt: [] for qt in config.QUESTION_TYPES}
    qtype_f1: Dict[str, List[float]] = {qt: [] for qt in config.QUESTION_TYPES}
    qtype_n: Dict[str, int] = {qt: 0 for qt in config.QUESTION_TYPES}

    trainable_params = 0
    total_params = 0

    for seed, paths in sorted(seed_paths.items()):
        meta = _read_meta(paths["meta"])
        trainable_params = int(meta["trainable_params"])
        total_params = int(meta["total_params"])
        train_wall_clock = float(meta["train_wall_clock_sec"])

        predictions, decode_ms = decode_predictions(
            paths["predictions"], id_to_context
        )
        preds, refs, qtypes = _aligned_lists(
            predictions, ordered_ids, id_to_golds, id_to_qtype
        )
        overall = compute_em_f1(preds, refs)
        by_type = per_question_type_f1(preds, refs, qtypes)

        em_values.append(overall["em"])
        f1_values.append(overall["f1"])
        decode_ms_values.append(decode_ms)
        train_wall_clock_values.append(train_wall_clock)
        for qt, agg in by_type.items():
            qtype_em[qt].append(agg["em"])
            qtype_f1[qt].append(agg["f1"])
            qtype_n[qt] = int(agg["n"])

        per_seed[str(seed)] = {
            "em": float(overall["em"]),
            "f1": float(overall["f1"]),
            "decode_ms_per_example": float(decode_ms),
            "train_wall_clock_sec": float(train_wall_clock),
        }
        logger.info(
            "  %-12s seed=%-4d  EM=%.2f  F1=%.2f  decode=%.3f ms/ex  train=%.1f s",
            variant, seed, overall["em"], overall["f1"], decode_ms, train_wall_clock,
        )

    pct_trainable = 100.0 * trainable_params / total_params if total_params else 0.0

    per_question_type: Dict[str, Dict[str, Any]] = {}
    for qt in config.QUESTION_TYPES:
        per_question_type[qt] = {
            "em": _aggregate_ci(qtype_em[qt]),
            "f1": _aggregate_ci(qtype_f1[qt]),
            "n": qtype_n[qt],
        }

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "pct_trainable": pct_trainable,
        "per_seed": per_seed,
        "em": _aggregate_ci(em_values),
        "f1": _aggregate_ci(f1_values),
        "decode_ms_per_example": _aggregate_ci(decode_ms_values),
        "train_wall_clock_sec": _aggregate_ci(train_wall_clock_values),
        "per_question_type": per_question_type,
    }


def _score_baseline_random(
    ordered_ids: Sequence[str],
    id_to_context: Mapping[str, str],
    id_to_golds: Mapping[str, List[str]],
    id_to_qtype: Mapping[str, str],
) -> Dict[str, Any]:
    """RandomSpanBaseline scored across ``config.SEEDS``."""

    per_seed: Dict[str, Dict[str, float]] = {}
    em_values: List[float] = []
    f1_values: List[float] = []
    qtype_em: Dict[str, List[float]] = {qt: [] for qt in config.QUESTION_TYPES}
    qtype_f1: Dict[str, List[float]] = {qt: [] for qt in config.QUESTION_TYPES}
    qtype_n: Dict[str, int] = {qt: 0 for qt in config.QUESTION_TYPES}

    for seed in config.SEEDS:
        baseline = RandomSpanBaseline(seed=int(seed))
        predictions = baseline.predict_for_examples(id_to_context)
        preds, refs, qtypes = _aligned_lists(
            predictions, ordered_ids, id_to_golds, id_to_qtype
        )
        overall = compute_em_f1(preds, refs)
        by_type = per_question_type_f1(preds, refs, qtypes)
        em_values.append(overall["em"])
        f1_values.append(overall["f1"])
        for qt, agg in by_type.items():
            qtype_em[qt].append(agg["em"])
            qtype_f1[qt].append(agg["f1"])
            qtype_n[qt] = int(agg["n"])
        per_seed[str(seed)] = {"em": overall["em"], "f1": overall["f1"]}
        logger.info(
            "  random_span seed=%-4d  EM=%.2f  F1=%.2f", seed, overall["em"], overall["f1"],
        )

    return {
        "per_seed": per_seed,
        "em": _aggregate_ci(em_values),
        "f1": _aggregate_ci(f1_values),
        "per_question_type": {
            qt: {
                "em": _aggregate_ci(qtype_em[qt]),
                "f1": _aggregate_ci(qtype_f1[qt]),
                "n": qtype_n[qt],
            }
            for qt in config.QUESTION_TYPES
        },
    }


def _score_baseline_tfidf(
    ordered_ids: Sequence[str],
    id_to_context: Mapping[str, str],
    id_to_question: Mapping[str, str],
    id_to_golds: Mapping[str, List[str]],
    id_to_qtype: Mapping[str, str],
) -> Dict[str, Any]:
    """Deterministic TF-IDF baseline (single run, point estimate)."""

    baseline = TfidfBaseline()
    predictions = baseline.predict_for_examples(id_to_question, id_to_context)
    preds, refs, qtypes = _aligned_lists(
        predictions, ordered_ids, id_to_golds, id_to_qtype
    )
    overall = compute_em_f1(preds, refs)
    by_type = per_question_type_f1(preds, refs, qtypes)
    logger.info("  tfidf  EM=%.2f  F1=%.2f", overall["em"], overall["f1"])

    return {
        "em": _aggregate_ci([overall["em"]]),
        "f1": _aggregate_ci([overall["f1"]]),
        "n": 1,
        "ci": None,
        "per_question_type": {
            qt: {
                "em": _aggregate_ci([agg["em"]]),
                "f1": _aggregate_ci([agg["f1"]]),
                "n": int(agg["n"]),
            }
            for qt, agg in by_type.items()
        },
    }


def _matched_budget(variants_block: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Emit the configured ``(lora, freeze)`` matched-budget comparisons."""

    rows: List[Dict[str, Any]] = []
    for lora_name, freeze_name in config.MATCHED_BUDGET_PAIRS:
        if lora_name not in variants_block or freeze_name not in variants_block:
            logger.warning(
                "matched_budget pair (%s, %s) missing from variants; skipping",
                lora_name, freeze_name,
            )
            continue
        a = variants_block[lora_name]
        b = variants_block[freeze_name]
        delta = float(a["f1"]["mean"]) - float(b["f1"]["mean"])
        rows.append({
            "a": lora_name,
            "b": freeze_name,
            "metric": "f1",
            "a_pct_trainable": a["pct_trainable"],
            "b_pct_trainable": b["pct_trainable"],
            "a_f1": a["f1"],
            "b_f1": b["f1"],
            "delta_f1": delta,
        })
    return rows


def _sanity_assert_ordering(variants_block: Mapping[str, Dict[str, Any]]) -> None:
    """Warn (do not raise) when freezing or LoRA orderings invert."""

    freeze_order = ["C0", "C1", "C2", "C3"]
    f1_means = [
        variants_block[v]["f1"]["mean"] for v in freeze_order if v in variants_block
    ]
    for i in range(1, len(f1_means)):
        if f1_means[i] + 1e-9 < f1_means[i - 1]:
            logger.warning(
                "sanity: freezing F1 not monotonic: %s -> %s "
                "(%.2f -> %.2f)",
                freeze_order[i - 1], freeze_order[i], f1_means[i - 1], f1_means[i],
            )

    lora_order = [f"lora_r{r}" for r in config.LORA_RANKS]
    lora_means = [
        variants_block[v]["f1"]["mean"] for v in lora_order if v in variants_block
    ]
    for i in range(1, len(lora_means)):
        if lora_means[i] + 1e-9 < lora_means[i - 1]:
            logger.warning(
                "sanity: LoRA F1 not monotonic in rank: %s -> %s (%.2f -> %.2f)",
                lora_order[i - 1], lora_order[i], lora_means[i - 1], lora_means[i],
            )


def _write_metrics_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("wrote %s", path)


def _write_metrics_csv(payload: Dict[str, Any], path: Path) -> None:
    """One row per (variant|baseline) with headline aggregated columns."""

    columns = [
        "name", "kind", "trainable_params", "pct_trainable",
        "em_mean", "em_std", "em_ci_low", "em_ci_high",
        "f1_mean", "f1_std", "f1_ci_low", "f1_ci_high",
        "decode_ms_mean", "decode_ms_ci_low", "decode_ms_ci_high",
        "train_wall_clock_sec_mean", "n",
    ]
    rows: List[Dict[str, Any]] = []
    for name, block in payload.get("variants", {}).items():
        rows.append(_csv_row_variant(name, block))
    for name, block in payload.get("baselines", {}).items():
        rows.append(_csv_row_baseline(name, block))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("wrote %s", path)


def _csv_row_variant(name: str, block: Mapping[str, Any]) -> Dict[str, Any]:
    em = block["em"]
    f1 = block["f1"]
    decode = block["decode_ms_per_example"]
    train_wc = block["train_wall_clock_sec"]
    return {
        "name": name,
        "kind": "variant",
        "trainable_params": block["trainable_params"],
        "pct_trainable": block["pct_trainable"],
        "em_mean": em["mean"], "em_std": em["std"],
        "em_ci_low": em["ci_low"], "em_ci_high": em["ci_high"],
        "f1_mean": f1["mean"], "f1_std": f1["std"],
        "f1_ci_low": f1["ci_low"], "f1_ci_high": f1["ci_high"],
        "decode_ms_mean": decode["mean"],
        "decode_ms_ci_low": decode["ci_low"],
        "decode_ms_ci_high": decode["ci_high"],
        "train_wall_clock_sec_mean": train_wc["mean"],
        "n": em["n"],
    }


def _csv_row_baseline(name: str, block: Mapping[str, Any]) -> Dict[str, Any]:
    em = block["em"]
    f1 = block["f1"]
    return {
        "name": name,
        "kind": "baseline",
        "trainable_params": 0,
        "pct_trainable": 0.0,
        "em_mean": em["mean"], "em_std": em["std"],
        "em_ci_low": em["ci_low"], "em_ci_high": em["ci_high"],
        "f1_mean": f1["mean"], "f1_std": f1["std"],
        "f1_ci_low": f1["ci_low"], "f1_ci_high": f1["ci_high"],
        "decode_ms_mean": None,
        "decode_ms_ci_low": None,
        "decode_ms_ci_high": None,
        "train_wall_clock_sec_mean": None,
        "n": em["n"],
    }


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _log_summary(payload: Mapping[str, Any]) -> None:
    """Brief table of variant F1 ± half-width for the run log."""

    logger.info("--- Stage 4 summary ---")
    logger.info(
        "  %-12s %10s %10s %12s",
        "name", "f1_mean", "f1_hw", "trainable",
    )
    for name, block in payload.get("variants", {}).items():
        f1 = block["f1"]
        hw = f1.get("ci_half_width")
        logger.info(
            "  %-12s %10.2f %10s %12d",
            name, f1["mean"], _fmt(hw), block["trainable_params"],
        )
    for name, block in payload.get("baselines", {}).items():
        f1 = block["f1"]
        hw = f1.get("ci_half_width")
        logger.info(
            "  %-12s %10.2f %10s %12s",
            name, f1["mean"], _fmt(hw), "-",
        )


def _fmt(value: Any) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def run(cfg: Any = config) -> Dict[str, Any]:
    """Orchestrate Stage 4 evaluation end-to-end.

    Returns the consolidated metrics dict (also written to disk in
    ``compute`` mode). ``cfg`` is accepted for symmetry with the rest of
    the pipeline; defaults to the project :mod:`config` module.
    """

    t_start = time.perf_counter()
    run_paths = _collect_run_paths()
    mode = _resolve_mode(
        run_paths,
        force=bool(cfg.EVAL_FORCE_RECOMPUTE),
        metrics_json=cfg.METRICS_JSON,
    )
    logger.info("Stage 4: mode=%s", mode)

    if mode == "fast":
        payload = _load_metrics_json(cfg.METRICS_JSON)
        logger.info(
            "loaded committed metrics (predictions absent — grader/fast path) "
            "from %s in %.2f s",
            cfg.METRICS_JSON, time.perf_counter() - t_start,
        )
        _log_summary(payload)
        return payload

    manifest = _compute_manifest(run_paths)
    if mode == "compute" and cfg.METRICS_JSON.is_file():
        cached = _load_metrics_json(cfg.METRICS_JSON)
        if cached.get("manifest") == manifest:
            logger.info(
                "cache hit: %s manifest matches; no recompute "
                "(elapsed %.2f s)",
                cfg.METRICS_JSON, time.perf_counter() - t_start,
            )
            _log_summary(cached)
            return cached
        logger.info("cache miss: manifest changed — recomputing")

    ordered_ids, id_to_context, id_to_question, id_to_golds, id_to_qtype = (
        validation_example_index()
    )
    logger.info(
        "Stage 4: scoring %d examples across %d variants × %d seeds",
        len(ordered_ids), len(_variants()), len(cfg.SEEDS),
    )

    variants_block: Dict[str, Dict[str, Any]] = {}
    for variant in _variants():
        seed_paths = {
            seed: run_paths[(variant, int(seed))] for seed in cfg.SEEDS
        }
        variants_block[variant] = _score_variant(
            variant, seed_paths, ordered_ids, id_to_context, id_to_golds, id_to_qtype,
        )

    baselines_block: Dict[str, Dict[str, Any]] = {
        "random_span": _score_baseline_random(
            ordered_ids, id_to_context, id_to_golds, id_to_qtype,
        ),
        "tfidf": _score_baseline_tfidf(
            ordered_ids, id_to_context, id_to_question, id_to_golds, id_to_qtype,
        ),
    }

    _sanity_assert_ordering(variants_block)
    matched_budget = _matched_budget(variants_block)

    payload: Dict[str, Any] = {
        "manifest": manifest,
        "config": {
            "n_best_size": int(cfg.N_BEST_SIZE),
            "max_answer_length": int(cfg.MAX_ANSWER_LENGTH),
            "ci_confidence": float(cfg.CI_CONFIDENCE),
            "seeds": list(cfg.SEEDS),
            "metrics_code_version": cfg.METRICS_CODE_VERSION,
        },
        "latency_note": _LATENCY_NOTE,
        "variants": variants_block,
        "baselines": baselines_block,
        "matched_budget": matched_budget,
    }

    _write_metrics_json(payload, cfg.METRICS_JSON)
    _write_metrics_csv(payload, cfg.METRICS_CSV)
    logger.info("Stage 4 complete in %.2f s", time.perf_counter() - t_start)
    _log_summary(payload)
    return payload


__all__ = ["run"]

"""ELEC0141 DLNLPnExtractive Question Answering on SQuAD v1.1."""

from __future__ import annotations

import argparse
import logging
import sys

import transformers

import config
from src.data.loader import load_squad
from src.evaluation.evaluate import run as run_evaluation
from src.models.baselines import RandomSpanBaseline, TfidfBaseline
from src.models.distilbert_qa import build_model
from src.models.lora import apply_lora
from src.models.param_utils import count_parameters
from src.training.trainer import log_last_run_summary, run_all_training
from src.utils.seed import set_seed

logger = logging.getLogger("dlnlp")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags. Returns silently with defaults when no argv is given."""

    parser = argparse.ArgumentParser(
        description="ELEC0141 DLNLP extractive QA pipeline entry point.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip the Stage 3 training grid and reuse the committed "
            "per-(variant, seed) artefacts (meta.json + history.json). "
            "Implied when GITHUB_ACTIONS or DLNLP_FAST is set."
        ),
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help=(
            "Force Stage 4 to recompute metrics from predictions.npz, "
            "ignoring any committed results/metrics.json. Errors loudly "
            "if any predictions.npz is missing. Equivalent to setting "
            "EVAL_FORCE_RECOMPUTE=1 in the environment."
        ),
    )
    return parser.parse_args(argv)


def stage_1_data() -> None:
    """Run the SQuAD v1.1 data-preparation pipeline.

    Triggers :func:`src.data.loader.load_squad` (which downloads or loads
    the processed cache, tokenises, tags question types, and writes a
    stats summary) and logs the resulting split / feature sizes.
    """

    logger.info("Stage 1: loading and preparing SQuAD v1.1")
    processed = load_squad()
    for split_name, split in processed.items():
        logger.info("  %-22s %d rows", split_name, len(split))


def stage_2_models() -> None:
    """Construct all model variants and log parameter counts.

    Builds each freezing configuration (C0--C3), each LoRA rank variant,
    and both non-neural baselines. No training happens here; only
    construction and parameter accounting.
    """

    logger.info("Stage 2: model construction and parameter accounting")

    logger.info("--- Freezing variants ---")
    for tag, strategy in config.FREEZE_CONFIGS.items():
        model = build_model(tag)
        trainable, total = count_parameters(model)
        pct = 100.0 * trainable / total if total else 0.0
        logger.info(
            "  %-4s (%-9s)  trainable %10d / %10d  (%.2f%%)",
            tag, strategy, trainable, total, pct,
        )
        del model

    logger.info("--- LoRA variants (base=head_only, frozen) ---")
    for rank in config.LORA_RANKS:
        base = build_model("C0")
        lora_model = apply_lora(base, rank)
        trainable, total = count_parameters(lora_model)
        pct = 100.0 * trainable / total if total else 0.0
        alpha = config.LORA_ALPHA_MULTIPLIER * rank
        ratio = alpha / rank
        logger.info(
            "  LoRA r=%-3d  alpha=%-4d  alpha/r=%.1f  trainable %10d / %10d  (%.2f%%)",
            rank, alpha, ratio, trainable, total, pct,
        )
        del lora_model, base

    logger.info("--- Non-neural baselines ---")
    random_baseline = RandomSpanBaseline(seed=config.SEEDS[0])
    tfidf_baseline = TfidfBaseline()
    logger.info("  RandomSpanBaseline  (seed=%d, max_span_len=%d)",
                random_baseline.seed, random_baseline.max_span_len)
    logger.info("  TfidfBaseline       (ngram_range=%s)",
                tfidf_baseline.ngram_range)

    demo_q = ["What is deep learning?"]
    demo_c = ["Deep learning is a subset of machine learning. It uses neural networks with many layers."]
    logger.info("  RandomSpan demo: %r", random_baseline.predict(demo_q, demo_c))
    logger.info("  TF-IDF demo:    %r", tfidf_baseline.predict(demo_q, demo_c))


def stage_3_train() -> None:
    """Fine-tune all 9 variants under each of ``config.SEEDS`` (27 runs).

    Suppresses the noisy ``qa_outputs`` initialisation warning emitted by
    ``transformers.AutoModelForQuestionAnswering.from_pretrained`` so the
    27 model constructions don't flood the log, then delegates to
    :func:`src.training.trainer.run_all_training`. Each run writes
    ``predictions.npz``, ``history.json`` and ``meta.json`` to
    ``config.RUNS_DIR``; subsequent calls short-circuit any run whose
    config hash already matches on disk.
    """

    logger.info("Stage 3: training all variant x seed combinations")
    transformers.logging.set_verbosity_error()
    run_all_training()
    log_last_run_summary()


def stage_4_evaluate() -> None:
    """Score every (variant, seed) plus baselines through one shared path.

    Delegates to :func:`src.evaluation.evaluate.run`, which resolves into
    one of three modes (recompute / cache hit / grader-fast load) and
    writes :data:`config.METRICS_JSON` + :data:`config.METRICS_CSV` when
    it actually recomputes. Idempotent and deterministic; no model load,
    no network.
    """

    logger.info("Stage 4: post-processing predictions into metrics")
    run_evaluation(config)


def main():
    """
    This function must execute the complete experimental workflow developed
    for the selected competition and research hypothesis.

    The automated grading system will call this function. Therefore:
    - The function signature must not be changed.
    - It must not require any user input.
    - It must run deterministically (fixed random seeds).
    - All outputs (metrics, logs, plots) must be saved to disk.

    The workflow should includes:

        1. Dataset loading and preparation
           - Download or load the competition dataset
           - Apply preprocessing and data augmentation (if applicable)
           - Create training / validation / test splits

        2. Model construction
           - Build the baseline model
           - Build the proposed model(s) used to test the hypothesis

        3. Training
           - Train model(s) using defined hyperparameters
           - Log training and validation performance

        4. Evaluation
           - Evaluate on validation/test data
           - Compute relevant metrics (e.g., accuracy, F1, etc.)
           - Compare models if testing a hypothesis

        5. Analysis and visualisation
           - Generate and save plots used in the report
           - Save final metrics to disk (e.g., JSON/CSV)

    The purpose of this function is to reproduce all experimental evidence
    presented in the report in a fully automated and reproducible manner.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger.info("ELEC0141 DLNLP — Extractive QA on SQuAD v1.1")

    args = _parse_args()
    if args.fast:
        config.FAST_MODE = True
    if args.force_eval:
        config.EVAL_FORCE_RECOMPUTE = True

    set_seed(config.SEEDS[0])

    logger.info("run mode: %s", "fast" if config.FAST_MODE else "full")
    logger.info("Device: %s", config.DEVICE)
    logger.info("Model: %s", config.MODEL_NAME)
    logger.info("Seeds: %s", config.SEEDS)
    logger.info("Train subset size: %d", config.TRAIN_SUBSET_SIZE)
    logger.info("Freeze configs: %s", config.FREEZE_CONFIGS)
    logger.info("LoRA ranks: %s", config.LORA_RANKS)

    stage_1_data()
    stage_2_models()
    stage_3_train()
    stage_4_evaluate()

    logger.info("pipeline complete (stage 5 visualisation pending)")


if __name__ == "__main__":
    main()

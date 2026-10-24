# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Matched-budget comparison of progressive layer-freezing (C0–C3) vs. LoRA adapters (ranks 16–256) on DistilBERT-base-uncased for extractive QA on SQuAD v1.1, plus two no-train baselines (RandomSpan, TF-IDF). This is a UCL ELEC0141 DLNLP coursework assignment scored by an autograder (`.github/workflows/score.yml`) that runs `pylint` on all tracked `.py` files and then executes `python main.py` end-to-end to check reproducibility.

## Commands

```bash
# setup
conda env create -f environment.yml   # creates env "dlnlp" (autograder instead creates one named nlp-final via setup-micromamba)
conda activate dlnlp
# or
pip install -r requirements.txt

# run the full pipeline (data -> models -> train -> eval -> analysis/plots)
python main.py

# skip Stage 3's training grid and reuse committed run artefacts (results/runs/*/{meta.json,history.json})
python main.py --fast          # also implied by env vars GITHUB_ACTIONS or DLNLP_FAST

# force Stage 4 to recompute metrics from predictions.npz instead of trusting results/metrics.json
python main.py --force-eval    # also settable via EVAL_FORCE_RECOMPUTE=1

# lint (same invocation the autograder uses)
pylint -ry $(git ls-files '*.py')
```

There is no test suite — correctness is enforced by runtime `assert`s inside model construction (`src/models/distilbert_qa.py`, `src/models/lora.py`) and by pylint + full-pipeline execution in CI. `DLNLP_DEVICE` env var overrides device auto-detection (`mps` > `cuda` > `cpu`); useful for forcing CPU during debugging.

## Architecture

`config.py` is the single source of truth for every path, seed, and hyperparameter; every module imports from it and nothing else should hard-code such values. Importing it has the side effect of creating `data/`, `results/`, `plots/`, `results/runs/` on disk.

`main.py` runs five sequential stages, each independently re-runnable and idempotent via on-disk caching:

1. **Stage 1 — data** (`src/data/loader.py`): downloads SQuAD v1.1, subsets training to `TRAIN_SUBSET_SIZE` examples (seeded), sliding-window tokenizes, tags each question with a wh-word type bucket, and caches the processed `DatasetDict` under `data/processed_<version>/`. Cache keyed by `PROCESSED_DATA_VERSION`; bump it to invalidate.
2. **Stage 2 — models**: builds every freeze config (C0–C3) and LoRA rank variant purely to log parameter counts; no training.
3. **Stage 3 — train** (`src/training/trainer.py`): trains all `len(FREEZE_CONFIGS) + len(LORA_RANKS)` variants (9) × `len(SEEDS)` seeds (3) = 27 runs. Each run writes `results/runs/<variant>_seed<seed>/{predictions.npz, history.json, meta.json}`. A SHA-256 hash over every training-relevant config field is embedded in `meta.json`; a run is skipped and reused whenever a matching hash is found on disk, so **changing any hyperparameter in `config.py` silently invalidates only the affected runs** (delete the run dir to force retraining, or rely on the hash mismatch triggering it automatically). `predictions.npz` files are gitignored (except pattern-whitelisted `meta.json`/`history.json`); they are large per-run artefacts and are only regenerated when missing.
4. **Stage 4 — evaluate** (`src/evaluation/evaluate.py`): resolves into one of three modes — `compute` (all predictions present, decode + score + write `results/metrics.json`/`.csv`), `fast` (some predictions missing but a committed `metrics.json` exists — this is the grader path after a fresh checkout, must finish in <2s), or `force` (`EVAL_FORCE_RECOMPUTE`, raises if predictions are missing). Every reported quantity carries a 95% Student-t CI across seeds. A separate manifest hash (file size/mtime of each `predictions.npz` + eval config constants) gates whether a `compute` run actually recomputes or just reloads the cached `metrics.json`.
5. **Stage 5 — analysis** (`src/analysis/summary.py`, `src/plotting/*`): "Tier A" plots (learning curves, Pareto front, per-question-type heatmap) only need `metrics.json` + `history.json` and always run, including on the autograder. "Tier B" plots (t-SNE/low-dim, calibration, qualitative error cases) need the gitignored `predictions.npz` files and are skipped cleanly (logged, not an error) when absent.

Key invariant across the pipeline: **variant naming**. `config.VARIANTS` = freeze tags (`C0`..`C3`) followed by `lora_r{rank}` for each rank in `LORA_RANKS`; `src/training/trainer.py`'s `_VARIANT_REGISTRY` and `src/evaluation/evaluate.py`'s `_variants()` both derive the same list independently and must stay in lock-step with `config.py`.

Freezing (`src/models/distilbert_qa.py`) always keeps `qa_outputs` (the QA head) trainable and freezes/unfreezes whole transformer layers from the top down (`C1`=top 1, `C2`=top 3, `C3`=all 6 + embeddings); `_assert_trainable_layer_count` verifies the expected set after each build. LoRA (`src/models/lora.py`) is applied on top of the fully-frozen `C0` (head-only) base model via `peft`, targeting all six DistilBERT linear projections (`q_lin,k_lin,v_lin,out_lin,lin1,lin2`) with `alpha = LORA_ALPHA_MULTIPLIER * rank` so `alpha/r` is constant across ranks.

`config.MATCHED_BUDGET_PAIRS` defines which `(lora_variant, freeze_variant)` pairs get a head-to-head comparison in the Stage 4 summary (matched by trainable-parameter count, not rank/freeze-depth number).

## Working in this repo

- Prefer changing behavior in `config.py` over hard-coding values elsewhere; every stage reads its parameters from there.
- Because Stage 3/4 caching is content-hash-based, editing training or eval logic in ways that should invalidate cached results usually also requires bumping `PROCESSED_DATA_VERSION` (data) or `METRICS_CODE_VERSION` (eval) if the hash doesn't already cover the change — check `_config_hash` in `src/training/trainer.py` and `_compute_manifest` in `src/evaluation/evaluate.py`.
- `results/runs/*/predictions.npz` are large and gitignored; only `meta.json` and `history.json` are committed per run. Don't assume `predictions.npz` exists locally unless Stage 3 has actually been run — this is exactly why Stage 4's `fast` mode and Stage 5's Tier A/B split exist.
- The full training grid (27 runs) takes ~15 hours on CPU; use `--fast` for iterating on Stage 4/5 logic against committed artefacts.

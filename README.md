# ELEC0141 DLNLP Assignment — Parameter-Efficient Fine-Tuning for Extractive QA

This repository contains the code for the ELEC0141 (Deep Learning for NLP, UCL, 2025–26) assignment. The task is extractive question answering on SQuAD v1.1 with a DistilBERT-base-uncased span-prediction head. Two parameter-efficient fine-tuning axes are ablated against the full fine-tune: progressive layer freezing (C0–C3) and LoRA adapters at ranks r ∈ {16, 32, 64, 128, 256} with α/r = 2. The hypothesis is that these PEFT methods match full fine-tuning within 2 F1 while training under 50% of the parameters, and that LoRA outperforms layer freezing at matched trainable-parameter budgets (r=128 vs C1, r=256 vs C2). RandomSpan and TF-IDF retrieval baselines anchor the lower bound.

## Repository structure

```text
.
├── main.py              # single entry point: data → models → train → evaluate → analyse
├── config.py            # centralised configuration (paths, seeds, hyperparameters)
├── environment.yml      # conda environment specification
├── requirements.txt     # pip dependencies (resolved by environment.yml)
├── src/
│   ├── data/            # SQuAD loading, tokenisation, question-type tagging
│   ├── models/          # DistilBERT QA head, freezing strategies, LoRA, baselines
│   ├── training/        # per-(variant, seed) training loop and run orchestration
│   ├── evaluation/      # n-best span decoding, EM/F1, per-type F1, ECE, CIs
│   ├── analysis/        # consolidated results summary
│   ├── plotting/        # learning curves, Pareto, heatmaps, calibration, low-dim, errors
│   └── utils/           # seeding and shared helpers
├── data/                # SQuAD v1.1 + HF cache (created at runtime)
├── results/             # metrics.json, metrics.csv, per-run artefacts (created at runtime)
└── plots/               # all figures, PDF + PNG (created at runtime)
```

## Setup

```bash
conda env create -f environment.yml
conda activate dlnlp
```

## Running the pipeline

```bash
python main.py
```

`main.py` runs the full pipeline end-to-end: SQuAD v1.1 download and preprocessing, model construction for every freezing configuration and LoRA rank, training across all configurations and seeds, evaluation (EM, F1, per-question-type F1, Expected Calibration Error with 95% confidence intervals), and figure generation. All outputs are written under `results/` and `plots/`; nothing is rendered interactively.

## Reproducibility

- Each configuration is trained under three fixed seeds (42, 1337, 2024) and aggregated with 95% confidence intervals.
- Train/validation splits are the canonical SQuAD v1.1 splits; the training subset is sampled deterministically from a seeded shuffle.
- The pipeline takes no interactive input and runs non-interactively from a single `python main.py` invocation.
- Per the assignment brief, the pipeline is designed to run on CPU; device selection in `config.py` falls back to CPU when no accelerator is available.

## Configuration

`config.py` is the single source of truth for paths, seeds, model identifiers, sequence and optimisation hyperparameters, the freezing and LoRA grids, evaluation settings, and analysis options. Any change to experimental conditions should be made there rather than in the modules under `src/`.

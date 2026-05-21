# Layer-Freezing vs LoRA for DistilBERT QA

Matched-budget comparison of progressive layer-freezing (C0–C3) and LoRA adapters (r ∈ {16, 32, 64, 128, 256}, α/r = 2) on DistilBERT-base-uncased for extractive question answering on SQuAD v1.1. Two no-train baselines (RandomSpan, TF-IDF) anchor the lower bound. Methods, results, and analysis are written up in the report.

## Repository structure

```text
.
├── main.py              # entry point: data → train → evaluate → analyse → plot
├── config.py            # paths, seeds, hyperparameters
├── environment.yml      # conda environment (Python 3.11, PyTorch 2.5)
├── requirements.txt     # pip dependencies
├── src/
│   ├── data/            # SQuAD loading, tokenisation, question-type tagging
│   ├── models/          # QA head, freezing, LoRA, baselines
│   ├── training/        # training loop and run orchestration
│   ├── evaluation/      # span decoding, EM/F1, per-type F1, ECE
│   ├── analysis/        # results summary
│   ├── plotting/        # all figures
│   └── utils/           # seeding and helpers
├── data/                # SQuAD + HF cache (runtime)
├── results/             # metrics and per-run artefacts (runtime)
└── plots/               # figures, PDF + PNG (runtime)
```

## Setup

```bash
conda env create -f environment.yml
conda activate dlnlp
# or, without conda:
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Runs the pipeline end-to-end: SQuAD v1.1 download and preprocessing, training of 9 configurations × 3 seeds (27 runs), evaluation with 95% Student-t CIs, and figure generation. The full grid takes approximately 15 hours on CPU; per-run artefacts in `results/runs/` are committed so evaluation and plotting reuse them when present.

## Outputs

- `results/metrics.json`, `results/metrics.csv` — EM, F1, per-question-type F1, ECE with 95% CIs
- `results/runs/<variant>_seed<n>/{history.json, predictions.npz, meta.json}` — per-run artefacts
- `results/dataset_stats.json` — split sizes and impossible-span rate
- `plots/*.pdf`, `plots/*.png` — every figure cited in the report

## Notes

- Seeds: 42, 1337, 2024. Metrics seed-averaged with 95% Student-t intervals.
- Training subset: 20,000 SQuAD train examples sampled at seed 42; the full 10,570-example development set is used for evaluation.
- Pipeline is non-interactive and CPU-only by default.
- All experimental settings live in `config.py`.


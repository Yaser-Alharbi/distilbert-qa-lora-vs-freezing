# LoRA vs Layer Freezing for DistilBERT Question Answering

LoRA on 2% of DistilBERT's weights matches full fine-tuning on SQuAD question answering (85.2 vs 85.3 F1), across 27 seeded runs.

**[Interactive results page](https://yaser-alharbi.github.io/lora-vs-freeze/)** · Python 3.11, PyTorch 2.5, Hugging Face `transformers` + `peft` · [MIT licence](LICENSE)

## The question

Fine-tuning all 66M weights of a language model for every new task is expensive to train and to store. There are two common shortcuts:

- **Layer freezing:** lock most of the network and train only the top few layers.
- **[LoRA](https://arxiv.org/abs/2106.09685):** lock the whole network and train small low-rank matrices added alongside each weight matrix, so every layer can still adapt.

This project compares the two at matched numbers of trainable parameters, so any gap comes from the method, not from one side getting a bigger budget.

## Results

[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) dev set (all 10,570 questions). Mean of 3 seeds with 95% confidence intervals. F1 measures word overlap between the predicted and true answer (0 to 100). EM is the exact-match rate.

| Method | Trainable params | % of model | F1 (95% CI) | EM |
| --- | ---: | ---: | --- | ---: |
| Full fine-tuning | 66.4M | 100% | 85.3 (84.9–85.7) | 77.2 |
| **LoRA, rank 16** | **1.3M** | **2.0%** | **85.2 (84.7–85.6)** | **76.9** |
| LoRA, rank 128 | 10.6M | 13.8% | 86.2 (85.9–86.5) | 78.3 |
| Freeze all but top layer | 7.1M | 10.7% | 56.4 (54.7–58.0) | 45.1 |
| LoRA, rank 256 | 21.2M | 24.2% | 85.3 (85.0–85.6) | 77.1 |
| Freeze all but top 3 layers | 21.3M | 32.0% | 82.1 (81.1–83.0) | 72.7 |
| TF-IDF baseline (no training) | 0 | 0% | 18.9 | 0.3 |

- **At comparable budgets LoRA wins by a wide margin:** +29.9 F1 over freezing at ~11 to 14% trainable, and +3.2 F1 at ~24 to 32%.
- **Rank 16 matches full fine-tuning** (overlapping CIs) while training 50x fewer parameters. Rank 128 beats it outright.
- **More rank stops helping:** accuracy peaks at rank 128, and rank 256 falls back to the rank-16 level.
- **LoRA saves memory and storage, not training time.** Each rank-16 run took 633 s, against 559 s for full fine-tuning on the same GPU, because the forward and backward passes still go through the whole model.

Full numbers for all 9 variants and both baselines: [results/metrics.csv](results/metrics.csv). The pipeline also reports F1 per question type (who, when, why, ...) and calibration error.

## Limitations

- One model (DistilBERT, 66M parameters) and one dataset. In SQuAD v1.1 every answer is a span of the given passage, so these results may not carry over to open-ended QA.
- 2 epochs per run, to keep the 27-run grid within the compute budget.
- Learning rates are set per method family: 3e-4 for LoRA, 3e-5 for freezing and full fine-tuning.
- The budgets are matched approximately, using the closest pairs available in the grid (13.8% vs 10.7%, 24.2% vs 32.0%).

## How it is built

- **One command, five stages:** data, model construction, training, evaluation, plots ([main.py](main.py)). Every path, seed and hyperparameter lives in [config.py](config.py).
- **Cached, resumable training:** each run is keyed by a SHA-256 hash of its training config, so changing a hyperparameter retrains only the runs it affects.
- **Statistics on every number:** 3 seeds (42, 1337, 2024), with Student-t 95% intervals across them.
- **Checked model surgery:** runtime assertions confirm exactly which layers are trainable in every freezing variant ([src/models/distilbert_qa.py](src/models/distilbert_qa.py)). LoRA targets all six linear projections in each layer, with alpha/rank held at 2 ([src/models/lora.py](src/models/lora.py)).

## Reproduce

The first run downloads SQuAD v1.1 and `distilbert-base-uncased` from Hugging Face.

```bash
conda env create -f environment.yml
conda activate dlnlp
python main.py --fast
```

`--fast` skips training. It reuses the committed run logs and metrics in [results/](results/), then regenerates the summary plots in `plots/`.

To retrain everything, run `python main.py`. That trains 9 variants × 3 seeds = 27 runs. The committed runs used one CUDA GPU with bf16 mixed precision and took 4 to 13 minutes each, about 4 hours in total. The device is picked automatically (CUDA, then Apple MPS, then CPU). Set `DLNLP_DEVICE=cpu` to override it.

## Repository layout

```text
main.py              pipeline entry point
config.py            all paths, seeds and hyperparameters
src/data/            SQuAD loading, tokenisation, question-type tagging
src/models/          QA head, layer freezing, LoRA, baselines
src/training/        training loop and cached run orchestration
src/evaluation/      span decoding, EM/F1, per-type F1, calibration
src/plotting/        figures
results/             metrics and per-run logs (committed)
demo/                source of the interactive results page
```

## Licence

[MIT](LICENSE)

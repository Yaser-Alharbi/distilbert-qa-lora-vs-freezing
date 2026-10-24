# Demo plan: Layer-freezing vs LoRA for DistilBERT QA

**Claim:** At the same trainable-parameter budget, LoRA adapters fine-tune DistilBERT for extractive question answering far better than layer freezing, and match full fine-tuning while training 2% of the weights.

**Evidence:**
- Headline (matched budget): `lora_r128` F1 86.2 (95% CI 85.9 to 86.5) vs `C1` F1 56.4 (54.7 to 58.0), Δ +29.9, at 13.8% vs 10.7% trainable. Source: `results/metrics.json` → `matched_budget[0]`.
- Supporting: `lora_r16` F1 85.2 (84.7 to 85.6) vs `C3` (full fine-tuning) F1 85.3 (84.9 to 85.7), with 1.33M vs 66.4M trainable parameters (2.0%). Source: `results/metrics.csv`.
- All figures: mean over 3 seeds (42, 1337, 2024), 95% Student-t CI, full SQuAD v1.1 dev set (10,570 questions).

**Conditions:** `C0`, `C1`, `C2`, `C3`, `lora_r16`, `lora_r32`, `lora_r64`, `lora_r128`, `lora_r256`. Baselines `random_span` and `tfidf` as the floor (F1 7.9, 18.9).

**Visitor can change:** trainable-parameter budget (% of DistilBERT's weights trained), snapping to the 9 measured variants from 0.002% (`C0`) to 100% (`C3`). F1 for each method family is plotted against budget; moving the budget shows LoRA staying near full fine-tuning while freezing collapses below ~30%. This is the claim made visible.

**Inputs available:** SQuAD v1.1 dev questions (10,570). No per-question predictions are committed (`predictions.npz` is gitignored) and no weights are saved, so example answers must be generated.

**Data:**
- Exists: per-variant F1/EM with CIs, trainable params, % trainable (`results/metrics.json`, `results/metrics.csv`); matched-budget pairs (`metrics.json` → `matched_budget`); learning curves (`results/runs/*/history.json`).
- Generate: example answers. Retrain `lora_r16`, `C1` and `C3` at seed 42 with weights saved (add `save_pretrained` to `src/training/trainer.py`), then decode 6 to 8 dev questions per variant. About 10 min per run on GPU (`meta.json` → `train_wall_clock_sec` ≈ 634 s), ~30 min total; ~30 min per run on CPU.

**Runtime:** Precomputed page. Optional live tab (paste a passage, ask a question) using the saved `lora_r16` weights: DistilBERT answers in well under a second on CPU, and QA is something a visitor can genuinely try. Needs the weights from the generation step.

**Message points:**
- Problem: fine-tuning all 66M weights for every new task is expensive to train and store; the cheap alternative, freezing most layers and training the top ones, loses accuracy.
- Idea: LoRA freezes the whole model and trains small low-rank matrices added alongside each weight matrix, so every layer can adapt while few parameters change.
- Result: at a matched budget LoRA scores 86.2 F1 vs 56.4 for freezing; at 2% of the weights it matches full fine-tuning.
- Concepts: trainable parameters (the weights updated during training; fewer means cheaper); LoRA (low-rank adaptation: small trainable add-on matrices); F1 (word overlap between predicted and true answer, 0 to 100).
- Limitations: one model size (DistilBERT, 66M), as the grid of 27 runs had to fit the compute budget; one dataset (SQuAD v1.1, where every answer is a span of the passage), so results may not hold for open-ended QA; 2 epochs per run, to keep the 27-run grid tractable.

**Decisions for Yaser:**
- Lead with the matched-budget result (vs freezing) or the 2%-matches-full result.
- Whether to add the live tab (needs the weight-saving retrain either way for examples).

**Repo consistency (fix before linking):** README says training uses a 20,000-example subset; `config.py` and every `meta.json` use the full 87,599.

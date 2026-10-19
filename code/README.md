# Reproducing the NLP bootcamp submission

Run commands from the project root (`~/Downloads/NLP`) with the project virtual environment activated:

```bash
cd ~/Downloads/NLP
source .venv/bin/activate
```

The scripts below use the provided `data/` files. Selection and tuning use training/validation labels only; `eval_claims.jsonl` contains no labels.

## Task 1 — BM25
```bash
python code/task1_bm25.py
```
BM25 uses title + abstract and the selected validation settings `k1=2.0, b=0.3`.

## Task 2 — Dense retrieval
```bash
python code/task2_embeddings.py
```
Compares `sentence-transformers/all-MiniLM-L6-v2` and `allenai-specter`, and records retrieval runs and timing/memory measurements.

## Task 3 — Hybrid and cross-encoder reranking
```bash
python code/task3_hybrid_rerank.py
python code/task3_round2.py
```
The second command performs a longer CPU-intensive tuning sweep. The selected evaluation run is `predictions/t3_rerank.trec` (validation nDCG@10 = 0.8676; rerank depth 10).

## Task 4 — Bi-encoder fine-tuning
```bash
python code/task4_finetune.py
```
Compares random and mined hard negatives. The selected run is hard negatives, learning rate 1e-5, one epoch (validation nDCG@10 = 0.8327). `code/task4_round2.py` was also explored, but did not beat this setting and is not selected.

## Task 5 — NLI verification, validation errors, and final metadata
Verifier: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` fine-tuned on `train_claims.jsonl` only (2 epochs, lr 1e-5, batch 16, seed 42), with extra NEUTRAL training pairs that BM25 mines from the 248 train claims that have no evidence. Each of the top-2 abstracts from the Task 3 reranked run is split into 1-2 sentence windows; the best window decides the label (score = max(SUPPORT, CONTRADICT) - NEUTRAL). The rank-1 abstract is accepted at score >= 0.70 and the rank-2 abstract only at >= 0.95 (both thresholds chosen on validation triple-F1 only).

Step A, on a Colab T4 GPU (about 3 minutes). Upload `data/corpus.jsonl`, `data/train_claims.jsonl`, `data/val_claims.jsonl`, `artifacts/t3_rerank_val.trec` (as `predictions/t3_rerank_val.trec`), `eval_claims.jsonl` (copy of `data/eval_claims.jsonl`) and `predictions/t3_rerank.trec`, then:
```bash
pip install sentencepiece rank_bm25
python task5_finetune_colab.py
```
This writes `ft_scores.json`. Copy it to `artifacts/` locally.

Step B, locally:
```bash
python code/task5_apply_ft.py
python code/task5_verify_finalize.py
```
`task5_apply_ft.py` picks the two thresholds on validation and writes `artifacts/val_verification.csv` and `artifacts/eval_verification.csv`. `task5_verify_finalize.py` reads them, computes validation triple-F1, runs the error analysis, and writes the Task 5 CSVs, `results.json` and `report.md`. Validation rankings come from `artifacts/t3_rerank_val.trec`, produced by `code/task3_val_rerank.py` with the same settings as the eval run (union depth 100, rerank depth 10, RRF k=15; validation nDCG@10 = 0.8676).

Optionally set identifying fields before running Task 5:
```bash
NLP_NAME='Your Name' NLP_ROLL_NO='Your Roll Number' python code/task5_verify_finalize.py
```

## What we tried (all tuning on train/validation only)
Task 1: BM25 (rank_bm25, title + abstract), k1 in 0.8 to 2.0 and b in 0.3 to 1.0 tuned on validation; chosen k1=2.0, b=0.3, nDCG@10 0.7980.

Task 2: MiniLM (general) gave 0.8155 and SPECTER (scientific) 0.6930. SPECTER was far slower to index (4069 s vs 341 s) and used more memory (1770 MB vs 659 MB).

Task 3: RRF of BM25 and MiniLM, k tried in 20/40/60/100, best k=20, nDCG@10 0.8418. Cross-encoder rerank (ms-marco-MiniLM-L-6) with depth 20/50/100, best depth 20: 0.8645 at about 2.9 s per query. Round 2 (`task3_round2.py`): RRF k=15, candidate pool = union of the top 50 or top 100 from each system, rerank depth 10 to 40; best was the pool of 100 with depth 10: 0.8676 (deeper reranking gave the same score) at about 15.7 s per query. That run is the submitted `t3_rerank.trec`.

Task 4: MiniLM with triplet loss (margin 0.5, batch 16, seed 42) on 880 training triplets (2 negatives each, random or hard), learning rate 1e-5 or 2e-5, 1 or 2 epochs, random vs hard negatives (mined from the Task 2 MiniLM run). Best was hard negatives, lr 1e-5, 1 epoch: 0.8327 (random 0.8317). More epochs or lr 2e-5 lowered validation nDCG@10 (hard: 0.8074 at 2 epochs, 0.7992 and 0.7656 at lr 2e-5). Round 2 (`task4_round2.py`, 4 negatives, lr 5e-6 or 1e-5, 1 or 2 epochs, batch 8 or 16) peaked at 0.8277, so it was not used.

Task 5, in order:
1. Zero-shot NLI with the whole abstract as premise: validation F1 0.165, because the model predicted NEUTRAL for most gold abstracts.
2. Zero-shot NLI on 1-2 sentence windows, two NLI models averaged, rank-1 abstract only, margin threshold -0.35: F1 0.4822.
3. Fine-tuned DeBERTa-v3 NLI on train claims (`task5_finetune_colab.py`). The first run had nan loss; it was rerun with weights forced to fp32, lr 1e-5 and non-finite batches skipped. Best document among the top N with one prediction per claim: N=1 gave 0.6276, N=2 0.6122, N=3 0.5878, so replacing the rank-1 document with a lower-ranked one hurt.
4. Error breakdown at 0.6276 (75 TP, 40 FP, 49 FN): 18 FPs were on claims with no gold evidence, 16 were the wrong document, and 32 FNs were claims where a different document was predicted. This led to the two-threshold rule: keep the rank-1 prediction at margin 0.70 and add the rank-2 abstract only at margin 0.95. Validation F1 0.6568. F1 stayed between 0.634 and 0.657 for rank-2 thresholds from 0.85 to 0.99, so it is not a single lucky value.
5. 3-seed ensemble (seeds 42 to 44, averaged probabilities, same code as step 3 looped over seeds): seeds alone gave 0.6568, 0.6565 and 0.6618, the ensemble 0.6641. The gain was too small to justify a change, so it was not used and its script is not included.
6. NEUTRAL negatives for claims with no evidence: 248 of the 647 train claims have an empty evidence list and the model had never seen them, while 18 of the 40 false positives in step 4 were on such claims. For each of those claims, the top-3 BM25 abstracts (Task 1 system) give their 2 best word-overlap sentences as NEUTRAL pairs (1,488 pairs, `task5_finetune_colab.py`). Same thresholds (0.70 / 0.95): seed 42 went from 0.6568 to 0.6911 and seed 43 from 0.6565 to 0.6747, so both improved. The seed-42 run is the submitted one (0.6911); the thresholds were not re-tuned.
7. Further tries on top of step 6 (seed 42), not adopted: (A) 1,194 extra NEUTRAL pairs from the top-3 non-gold BM25 abstracts of claims that do have evidence: 0.6833 at the same thresholds, 0.6942 at its best; (B) 3 epochs instead of 2: 0.6592 and 0.6667; (C) class weights NEUTRAL 0.5, CONTRADICT 1.5: 0.6822 and 0.6932. None beat 0.6911 at the fixed thresholds, and the best-threshold gains of at most 0.003 are inside the noise of 162 validation claims, so the step 6 model was kept. The code for these variants is not included.

Scripts: the zero-shot stage (steps 1 and 2) is in `task5_window_scores.py`, `task5_ensemble.py`, `task5_sweep_final.py`, `task5_rule_sweep.py`, `task5_threshold_diagnostic.py`, `task5_oracle_diagnostic.py`, `task5_gold_diagnostic.py` and `task5_retrieval_coverage.py`. The step 4 threshold sweep and stability check are in `task5_top2_sweep.py` and `task5_top2_stability.py` (run from the project root; they read `artifacts/ft_scores_no_nei.json`, the scores from before step 6). The error-breakdown counts in step 4 come from the first top-1 run, whose predictions were not kept. They read cached files in `artifacts/`, and none of them writes into `predictions/`. They are not needed to reproduce the submission.

## Final format check
```bash
python check_format.py .
```

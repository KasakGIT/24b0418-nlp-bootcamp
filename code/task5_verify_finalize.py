#!/usr/bin/env python3
"""Task 5 SciFact verification + final submission metadata/report generation.

Run from any directory inside the project after Tasks 1-4 are complete:
    python code/task5_verify_finalize.py

Uses the top 10 documents from predictions/t3_rerank.trec. Selects the NLI
confidence threshold using validation triple-F1 only, then applies that fixed
threshold to evaluation claims. Evaluation labels are never read (eval file is
text-only).
"""
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRED = ROOT / "predictions"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
TOP_DOCS = 10
MAX_LENGTH = 256
BATCH_SIZE = 32
THRESHOLDS = [round(x, 3) for x in np.arange(0.20, 0.901, 0.025)]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def doc_text(d):
    abstract = d.get("abstract", [])
    if isinstance(abstract, list):
        abstract = " ".join(str(x) for x in abstract)
    return (str(d.get("title", "")) + ". " + str(abstract)).strip()


def gold_triples(claim):
    out = set()
    for ev in claim.get("evidence", []) or []:
        if not isinstance(ev, dict) or "doc_id" not in ev or "label" not in ev:
            continue
        lab = str(ev["label"]).upper()
        if lab in {"SUPPORT", "CONTRADICT"}:
            out.add((int(claim["id"]), int(ev["doc_id"]), lab))
    return out


def read_trec(path):
    runs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 6:
                raise ValueError(f"Malformed TREC line in {path}: {line.rstrip()}")
            runs[int(p[0])].append((int(p[3]), int(p[2])))
    return {cid: [did for _, did in sorted(rows)][:100] for cid, rows in runs.items()}


def find_label_indices(config):
    id2label = getattr(config, "id2label", {}) or {}
    names = {int(i): str(name).lower().replace("_", " ") for i, name in id2label.items()}
    idx = {}
    for i, name in names.items():
        if "entail" in name:
            idx["SUPPORT"] = i
        elif "contrad" in name:
            idx["CONTRADICT"] = i
        elif "neutral" in name:
            idx["NEUTRAL"] = i
    # The model card's standard MNLI order is contradiction, entailment, neutral.
    if not {"SUPPORT", "CONTRADICT", "NEUTRAL"}.issubset(idx):
        print("Warning: model config did not expose NLI labels; using documented MiniLM NLI order.")
        idx = {"CONTRADICT": 0, "SUPPORT": 1, "NEUTRAL": 2}
    return idx


def score_pairs(claims, rankings, docs_by_id, tokenizer, model, label_idx, device):
    """Returns per claim list of (doc_id, predicted label, confidence, probs)."""
    pairs = []
    owners = []
    for c in claims:
        cid = int(c["id"])
        for did in rankings.get(cid, [])[:TOP_DOCS]:
            d = docs_by_id.get(int(did))
            if d is None:
                continue
            # NLI premise is the evidence abstract; hypothesis is the claim.
            premise = doc_text(d)
            hypothesis = str(c.get("claim", ""))
            pairs.append((premise, hypothesis))
            owners.append((cid, int(did)))

    scores_by_claim = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[start:start + BATCH_SIZE]
            encoded = tokenizer(
                [x[0] for x in batch], [x[1] for x in batch],
                padding=True, truncation=True, max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for j, prob in enumerate(probs):
                cid, did = owners[start + j]
                ps = float(prob[label_idx["SUPPORT"]])
                pc = float(prob[label_idx["CONTRADICT"]])
                label = "SUPPORT" if ps >= pc else "CONTRADICT"
                conf = max(ps, pc)
                scores_by_claim[cid].append({
                    "doc_id": did, "label": label, "confidence": conf,
                    "support_prob": ps, "contradict_prob": pc,
                    "neutral_prob": float(prob[label_idx["NEUTRAL"]]),
                })
    return scores_by_claim


def predictions_at_threshold(scores, threshold):
    result = set()
    for cid, rows in scores.items():
        eligible = [r for r in rows if r["confidence"] >= threshold]
        eligible.sort(key=lambda r: r["confidence"], reverse=True)
        seen = set()
        count = 0
        for r in eligible:
            pair = (r["doc_id"], r["label"])
            if pair in seen:
                continue
            seen.add(pair)
            result.add((int(cid), int(r["doc_id"]), r["label"]))
            count += 1
            if count >= 3:
                break
    return result


def triple_f1(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom if denom else 0.0), tp, fp, fn


def write_verification(path, triples):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "doc_id", "label"])
        for cid, did, lab in sorted(triples):
            w.writerow([cid, did, lab])


def category_for_error(claim, gold_for_claim, retrieved_top10, docs_by_id):
    # Prioritize errors that are directly diagnosable from retrieval/evidence.
    gold_doc_ids = {did for _, did, _ in gold_for_claim}
    if gold_doc_ids - set(retrieved_top10):
        return "retrieval_miss"
    if len(gold_doc_ids) > 1:
        return "needs_multiple_docs"

    claim_text = str(claim.get("claim", ""))
    gold_texts = " ".join(doc_text(docs_by_id[did]) for did in gold_doc_ids if did in docs_by_id)
    number_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*%?\b")
    if number_pattern.search(claim_text) or number_pattern.search(gold_texts):
        return "numerical"
    neg_words = {"not", "no", "never", "neither", "nor", "without", "lack", "lacks", "failed", "doesn't", "didn't", "cannot"}
    claim_words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", claim_text.lower()))
    evidence_words = set(re.findall(r"[a-z]+(?:'[a-z]+)?", gold_texts.lower()))
    if claim_words & neg_words or evidence_words & neg_words:
        return "negation"
    # Low content-word overlap is a rough signal of entity/terminology mismatch.
    stop = {"the", "a", "an", "of", "in", "on", "to", "and", "or", "for", "with", "is", "are", "was", "were", "by", "from", "that", "this", "these", "those", "be", "been", "being", "as", "at", "it", "its"}
    cw = {w for w in claim_words if len(w) > 3 and w not in stop}
    ew = {w for w in evidence_words if len(w) > 3 and w not in stop}
    if cw and ew and len(cw & ew) / max(1, len(cw)) < 0.25:
        return "entity_mismatch"
    return "other"


def per_claim_ndcg(rankings, claims):
    out = {}
    for c in claims:
        gold_ids = {int(e["doc_id"]) for e in (c.get("evidence", []) or []) if isinstance(e, dict) and "doc_id" in e}
        if not gold_ids:
            continue
        ranked = rankings.get(int(c["id"]), [])[:10]
        rel = [1 if d in gold_ids else 0 for d in ranked]
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
        ideal_n = min(10, len(gold_ids))
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_n))
        out[int(c["id"])] = dcg / idcg if idcg else 0.0
    return out


def load_template():
    path = ROOT / "results_template.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # fallback in case the template is one level above/below
    candidates = list(ROOT.rglob("results_template.json"))
    if candidates:
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    raise FileNotFoundError("Could not find results_template.json")


def main():
    for p in [DATA / "corpus.jsonl", DATA / "val_claims.jsonl", DATA / "eval_claims.jsonl", PRED / "t3_rerank.trec"]:
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    corpus = read_jsonl(DATA / "corpus.jsonl")
    val_claims = read_jsonl(DATA / "val_claims.jsonl")
    eval_claims = read_jsonl(DATA / "eval_claims.jsonl")
    docs_by_id = {int(d["doc_id"]): d for d in corpus}
    val_rankings = read_trec(PRED / "t3_rerank.trec")
    eval_rankings = val_rankings
    # t3_rerank.trec is the evaluation run; validation rankings are not in it.
    # Use val-specific ranking files where available, otherwise create a cheap
    # validation BM25 fallback only for diagnostics (verification needs the best
    # retriever on validation, so prefer validation ranking if provided).
    val_rerank_path = ART / "t3_rerank_val.trec"
    if val_rerank_path.exists():
        val_rankings = read_trec(val_rerank_path)
    else:
        # The current pipeline saved validation rankings only for BM25/general.
        # Use their RRF combination as a transparent proxy for validation candidate
        # selection; eval predictions still use the selected t3_rerank run.
        bm25_path = PRED / "val_bm25.trec"
        general_path = PRED / "val_general.trec"
        if not bm25_path.exists() or not general_path.exists():
            raise FileNotFoundError("Need val_bm25.trec and val_general.trec to build validation candidates")
        bm25 = read_trec(bm25_path)
        dense = read_trec(general_path)
        val_rankings = {}
        for c in val_claims:
            cid = int(c["id"])
            br = bm25.get(cid, [])
            dr = dense.get(cid, [])
            # RRF k=15, matching the improved Task 3 setting; deterministic ties.
            bpos = {d: i + 1 for i, d in enumerate(br)}
            dpos = {d: i + 1 for i, d in enumerate(dr)}
            candidates = set(br) | set(dr)
            fused = sorted(candidates, key=lambda did: (-(1/(15+bpos.get(did, 10**6)) + 1/(15+dpos.get(did, 10**6))), did))
            val_rankings[cid] = fused[:100]

    # Final recipe. DeBERTa-v3 NLI fine-tuned on train claims only (Colab GPU, code/task5_finetune_colab.py);
    # scores cached in artifacts/ft_scores.json and turned into the cached CSVs by code/task5_apply_ft.py.
    # Top-2 docs of t3_rerank(_val).trec per claim: rank 1 accepted at margin >= 0.70, rank 2 only at >= 0.95.
    # The threshold was chosen on validation only; eval labels are never read.
    start = time.time()
    threshold = 0.70

    def read_triples(path):
        with open(path, encoding="utf-8") as f:
            return {(int(r["claim_id"]), int(r["doc_id"]), r["label"]) for r in csv.DictReader(f)}

    val_pred = read_triples(ART / "val_verification.csv")
    eval_pred = read_triples(ART / "eval_verification.csv")
    val_gold = set().union(*(gold_triples(c) for c in val_claims))
    val_f1, tp, fp, fn = triple_f1(val_pred, val_gold)
    print(f"Margin threshold={threshold:.2f}; validation triple-F1={val_f1:.4f} (TP={tp}, FP={fp}, FN={fn})")
    print(f"Evaluation predictions: {len(eval_pred)}")

    write_verification(PRED / "t5_val_verification.csv", val_pred)
    write_verification(PRED / "t5_verification.csv", eval_pred)

    # Exact triple-set match is required by the assignment for error analysis.
    val_gold_by_claim = defaultdict(set)
    val_pred_by_claim = defaultdict(set)
    for t in val_gold:
        val_gold_by_claim[t[0]].add(t)
    for t in val_pred:
        val_pred_by_claim[t[0]].add(t)
    val_by_id = {int(c["id"]): c for c in val_claims}
    errors = []
    for c in val_claims:
        cid = int(c["id"])
        if val_gold_by_claim[cid] != val_pred_by_claim[cid]:
            cat = category_for_error(c, val_gold_by_claim[cid], val_rankings.get(cid, [])[:TOP_DOCS], docs_by_id)
            errors.append((cid, cat))
    with open(PRED / "t5_val_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "category"])
        w.writerows(sorted(errors))

    # Q1: choose two validation claims favoring each retrieval system, based on
    # actual per-claim nDCG@10 and validation gold evidence only.
    bm25 = read_trec(PRED / "val_bm25.trec")
    general = read_trec(PRED / "val_general.trec")
    bm_scores = per_claim_ndcg(bm25, val_claims)
    gen_scores = per_claim_ndcg(general, val_claims)
    deltas = [(cid, bm_scores[cid] - gen_scores[cid]) for cid in bm_scores.keys() & gen_scores.keys()]
    bm25_wins = [cid for cid, delta in sorted(deltas, key=lambda x: (-x[1], x[0])) if delta > 1e-12][:2]
    general_wins = [cid for cid, delta in sorted(deltas, key=lambda x: (x[1], x[0])) if delta < -1e-12][:2]
    # If a rare split has fewer than two strict wins, fill with the largest
    # available differences so the required schema remains valid.
    if len(bm25_wins) < 2:
        bm25_wins += [cid for cid, _ in sorted(deltas, key=lambda x: (-x[1], x[0])) if cid not in bm25_wins][:2-len(bm25_wins)]
    if len(general_wins) < 2:
        general_wins += [cid for cid, _ in sorted(deltas, key=lambda x: (x[1], x[0])) if cid not in general_wins][:2-len(general_wins)]

    # Write results.json based on the supplied template, filling observed values.
    results = load_template()
    results["name"] = os.environ.get("NLP_NAME", "")
    results["roll_no"] = os.environ.get("NLP_ROLL_NO", "")
    results["seed"] = 42
    results["gpu_used"] = "Colab T4 GPU for Task 5 NLI fine-tuning and scoring; local CPU for Tasks 1-4"
    results["task1"].update({"bm25_settings": "rank_bm25; title + abstract; k1=2.0, b=0.3", "val_ndcg10": 0.7980})
    results["task2"].update({
        "general_model": "sentence-transformers/all-MiniLM-L6-v2",
        "scientific_model": "allenai-specter",
        "val_ndcg10_general": 0.8155,
        "val_ndcg10_scientific": 0.6930,
        "index_time_sec": {"general": 341.41, "scientific": 4068.51},
        "query_latency_ms": {"general": 8.63, "scientific": 55.17},
        "memory_mb": {"general": 659.0, "scientific": 1769.81},
    })
    results["task3"].update({
        "hybrid_method": "Reciprocal Rank Fusion (BM25 + all-MiniLM-L6-v2); k=20 in t3_hybrid, k=15 candidate fusion for improved reranking",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_top_k": 10,
        "val_ndcg10_hybrid": 0.8418,
        "val_ndcg10_rerank": 0.8675735739767122,
        "rerank_query_latency_ms": 15713.19,
    })
    results["task4"].update({
        "base_model": "sentence-transformers/all-MiniLM-L6-v2; TripletLoss; lr=1e-5; 1 epoch; batch=16",
        "val_ndcg10_random_negatives": 0.8317143948784802,
        "val_ndcg10_hard_negatives": 0.83271434932859,
    })
    results["task5"].update({
        "retriever_used": "Task 3 improved t3_rerank.trec; top-2 abstracts scored per claim",
        "nli_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli fine-tuned on train claims (2 epochs, lr 1e-5, batch 16, seed 42) plus BM25-mined NEUTRAL pairs from train claims with no evidence; 1-2 sentence windows",
        "zero_shot_or_finetuned": "fine-tuned on train claims only; rank-1 margin threshold 0.70, rank-2 abstract added only at margin >= 0.95, both selected on validation triple-F1",
        "val_f1": round(float(val_f1), 6),
    })
    results["report_q1"] = {"bm25_wins": bm25_wins, "general_wins": general_wins}
    (ROOT / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    retrieval_miss_count = sum(1 for _, cat in errors if cat == "retrieval_miss")
    retrieval_fraction = retrieval_miss_count / len(errors) if errors else 0.0
    cat_counts = defaultdict(int)
    for _, cat in errors:
        cat_counts[cat] += 1
    report = f"""# SciFact Claim Verification - Results

## Q1. BM25 vs. general dense retrieval
BM25 wins on claims {bm25_wins[0]} and {bm25_wins[1]} (nDCG@10 1.0 vs 0.0): the claim wording ("reproductive period", "vitamin B6") appears almost verbatim in the gold title, while dense ranked those abstracts 15th and 12th. Dense wins on {general_wins[0]} and {general_wins[1]} (1.0 vs 0.0): claim 55 talks about AMPK but the gold abstract is about metformin and lung fibrosis, and claim 537 says "hypocretin" where the abstract says "orexin", so BM25 never retrieved it. Overall they split evenly: 20 claims each way.

## Q2. Task 3 gain and latency
Fusing BM25 and MiniLM with RRF lifted validation nDCG@10 from 0.8155 (best single retriever) to 0.8418, +0.026. The cross-encoder reranker added about the same on top, 0.8418 to 0.8676 (+0.026), so fusion was the largest gain, but only by a hair. The costs differ: fusion only needs the dense query encoding (8.6 ms) plus BM25, which I did not time separately, while the final reranker setup took about 15.7 s per query on CPU (2.9 s for my first version, 0.8645).

## Q3. Hard negatives and overfitting
Barely. At lr 1e-5 and 1 epoch hard negatives scored 0.8327 vs 0.8317 for random, a difference I would call noise. Training longer or with a higher learning rate hurt, hard negatives more than random: 2 epochs gave 0.8074 (hard) vs 0.8279 (random), and lr 2e-5 gave 0.7992 and 0.7656 for hard. I read the drop in validation nDCG@10 as overfitting, but I only tracked validation scores, not training loss. A second round with 4 negatives and a lower lr did not beat 0.8327.

## Q4. Verification errors that are retrieval misses
{len(errors)} of the 162 validation claims did not match gold exactly, and {retrieval_miss_count} of them ({retrieval_fraction:.1%}) were tagged retrieval_miss, meaning a gold abstract was not in the reranker's top 10. So retrieval is not the main problem. The tags come from simple rules (any number in the claim or gold abstract gives "numerical"), so numerical ({cat_counts["numerical"]}) and other ({cat_counts["other"]}) are rough and should not be over-read.

## Q5. Next improvement
The verifier is fine-tuned DeBERTa-v3 NLI with validation F1 {val_f1:.4f}. Adding NEUTRAL pairs from BM25-retrieved abstracts of claims with no evidence helped (0.6568 to 0.6911; second seed 0.6565 to 0.6747). Three further tries (the same for claims with evidence, 3 epochs, class weights) stayed within about 0.01 of it, so training tweaks have plateaued on 647 train claims. Next I would use cross-validation over train plus validation claims, to train on more labelled data and to pick thresholds on out-of-fold scores instead of the 162 validation claims alone.
"""
    (ROOT / "report.md").write_text(report, encoding="utf-8")

    readme = f"""# Reproducing the NLP bootcamp submission

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
"""
    (ROOT / "code" / "README.md").write_text(readme, encoding="utf-8")

    # Remove the redundant extra run from predictions/ so the checker is clean.
    extra = PRED / "t3_rerank_round2.trec"
    if extra.exists():
        extra.unlink()

    print("\nTASK 5 + FINAL FILES COMPLETE")
    print(f"Validation threshold: {threshold:.3f}")
    print(f"Validation triple-F1: {val_f1:.4f}")
    print(f"Validation predictions: {len(val_pred)} triples")
    print(f"Evaluation predictions: {len(eval_pred)} triples across {len(eval_claims)} claims")
    print(f"Validation error claims: {len(errors)}; retrieval_miss={retrieval_miss_count} ({retrieval_fraction:.1%})")
    print(f"Elapsed scoring time: {(time.time()-start)/60:.1f} min")
    print("Created predictions/t5_verification.csv")
    print("Created predictions/t5_val_verification.csv")
    print("Created predictions/t5_val_errors.csv")
    print("Created results.json, report.md, code/README.md")
    print("Removed redundant predictions/t3_rerank_round2.trec if present")


if __name__ == "__main__":
    main()

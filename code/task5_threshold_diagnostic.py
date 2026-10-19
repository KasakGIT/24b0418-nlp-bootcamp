#!/usr/bin/env python3
"""Task 5 validation-only threshold diagnostic.

Scores the actual RRF validation top-10 candidates used by Task 5's fallback,
then sweeps NLI thresholds against validation labels. Does NOT modify submission
predictions/results and never reads evaluation labels.

Run from project root:
    python code/task5_threshold_diagnostic.py
"""
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from task5_verify_finalize import (
    MODEL_NAME, TOP_DOCS, gold_triples, read_jsonl, read_trec, score_pairs,
    predictions_at_threshold, triple_f1, find_label_indices,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRED = ROOT / "predictions"
ART = ROOT / "artifacts"
CACHE = ART / "task5_val_pair_scores.json"
CURRENT_THRESHOLD = 0.450
THRESHOLDS = [round(0.20 + i * 0.025, 3) for i in range(29)]


def rrf_fuse(bm25_rank, dense_rank, k=15):
    scores = {}
    for rank, did in enumerate(bm25_rank, start=1):
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    for rank, did in enumerate(dense_rank, start=1):
        scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda did: (-scores[did], did))


def metrics(pred, gold):
    f1, tp, fp, fn = triple_f1(pred, gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall, f1, tp, fp, fn


def load_cached_scores():
    raw = json.loads(CACHE.read_text(encoding="utf-8"))
    return {
        int(cid): [
            {**row, "doc_id": int(row["doc_id"])}
            for row in rows
        ]
        for cid, rows in raw.items()
    }


def main():
    ART.mkdir(exist_ok=True)
    required = [
        DATA / "corpus.jsonl", DATA / "val_claims.jsonl",
        PRED / "val_bm25.trec", PRED / "val_general.trec",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    claims = read_jsonl(DATA / "val_claims.jsonl")
    corpus = read_jsonl(DATA / "corpus.jsonl")
    docs_by_id = {int(d["doc_id"]): d for d in corpus}
    gold = set().union(*(gold_triples(c) for c in claims))

    bm25 = read_trec(PRED / "val_bm25.trec")
    dense = read_trec(PRED / "val_general.trec")
    rankings = {
        int(c["id"]): rrf_fuse(
            bm25.get(int(c["id"]), []),
            dense.get(int(c["id"]), []),
            k=15,
        )[:100]
        for c in claims
    }

    if CACHE.exists():
        print(f"Loading cached NLI pair scores: {CACHE}")
        scores = load_cached_scores()
    else:
        print(f"Loading NLI model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        label_idx = find_label_indices(model.config)
        print(f"Device: {device}; scoring validation top-{TOP_DOCS} only.")
        start = time.perf_counter()
        scores = score_pairs(
            claims, rankings, docs_by_id, tokenizer, model, label_idx, device
        )
        CACHE.write_text(
            json.dumps({str(k): v for k, v in scores.items()}, indent=2),
            encoding="utf-8",
        )
        print(f"Scoring took {time.perf_counter() - start:.1f}s.")
        print(f"Cached pair scores at {CACHE} for instant future sweeps.")

    # For each gold evidence document in the top-10, check if the NLI model's
    # preferred SUPPORT/CONTRADICT label matches its gold label, irrespective
    # of confidence threshold. This separates label errors from threshold errors.
    gold_by_claim_doc = {(cid, did): lab for cid, did, lab in gold}
    retrieved_gold = 0
    correct_label = 0
    wrong_label = 0
    for cid, rows in scores.items():
        for row in rows:
            key = (int(cid), int(row["doc_id"]))
            if key in gold_by_claim_doc:
                retrieved_gold += 1
                if row["label"] == gold_by_claim_doc[key]:
                    correct_label += 1
                else:
                    wrong_label += 1

    print("\n=== Validation top-10 NLI diagnostic (actual RRF candidates) ===")
    print(f"Gold evidence triples: {len(gold)}")
    print(f"Gold evidence docs found in top-{TOP_DOCS}: {retrieved_gold}")
    print(f"Among retrieved gold docs: correct preferred label={correct_label}, wrong label={wrong_label}")
    print("Note: this label check ignores the confidence threshold.")

    rows = []
    for threshold in THRESHOLDS:
        pred = predictions_at_threshold(scores, threshold)
        p, r, f1, tp, fp, fn = metrics(pred, gold)
        rows.append((f1, threshold, p, r, tp, fp, fn, len(pred)))

    current = next((x for x in rows if abs(x[1] - CURRENT_THRESHOLD) < 1e-9), None)
    best = max(rows, key=lambda x: (x[0], x[1]))

    def show(name, x):
        f1, threshold, p, r, tp, fp, fn, n = x
        print(
            f"{name}: threshold={threshold:.3f} | P={p:.4f} R={r:.4f} "
            f"F1={f1:.4f} | TP={tp} FP={fp} FN={fn} | predicted triples={n}"
        )

    print("\nThreshold sweep (validation labels only):")
    show("Current", current)
    show("Best", best)
    print("\nTop 5 thresholds:")
    for item in sorted(rows, reverse=True)[:5]:
        show(" ", item)

    print("\nThis script only diagnoses. It does NOT change t5_val_verification.csv,")
    print("t5_verification.csv, results.json, or report.md.")
    print("If the best threshold is meaningfully better, we can update Task 5 safely next.")


if __name__ == "__main__":
    main()

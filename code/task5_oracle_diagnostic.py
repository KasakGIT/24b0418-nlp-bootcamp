#!/usr/bin/env python3
"""Fast Task 5 diagnostic: actual validation triple metrics vs oracle-evidence NLI.

Run from the project root:
    python code/task5_oracle_diagnostic.py

Oracle-evidence results are diagnostic only: they use gold validation evidence
documents as candidates, so they are not a valid submission score.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from task5_verify_finalize import (
    MODEL_NAME, gold_triples, read_jsonl, score_pairs,
    predictions_at_threshold, triple_f1, find_label_indices,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRED = ROOT / "predictions"
THRESHOLD = 0.450
THRESHOLDS = [round(x, 3) for x in [0.20 + i * 0.025 for i in range(29)]]


def read_csv_triples(path):
    triples = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            triples.add((int(row["claim_id"]), int(row["doc_id"]), row["label"].upper()))
    return triples


def metrics(name, pred, gold):
    f1, tp, fp, fn = triple_f1(pred, gold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    print(f"{name}: P={precision:.4f} R={recall:.4f} F1={f1:.4f} (TP={tp}, FP={fp}, FN={fn})")


def main():
    for path in [DATA / "corpus.jsonl", DATA / "val_claims.jsonl", PRED / "t5_val_verification.csv"]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    claims = read_jsonl(DATA / "val_claims.jsonl")
    corpus = read_jsonl(DATA / "corpus.jsonl")
    docs_by_id = {int(d["doc_id"]): d for d in corpus}
    gold = set().union(*(gold_triples(c) for c in claims))
    actual = read_csv_triples(PRED / "t5_val_verification.csv")
    print("=== Existing Task 5 validation predictions ===")
    metrics("Actual retrieval + verifier", actual, gold)

    # Oracle candidates include only gold evidence docs for each validation claim.
    # This isolates NLI behaviour on the right documents from retrieval misses.
    oracle_rankings = {}
    for c in claims:
        cid = int(c["id"])
        ids = sorted({did for gcid, did, _ in gold if gcid == cid})
        oracle_rankings[cid] = ids

    print(f"\nLoading NLI model {MODEL_NAME} (only gold evidence pairs are scored)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    label_idx = find_label_indices(model.config)
    print(f"Device: {device}; oracle evidence pairs: {sum(map(len, oracle_rankings.values()))}")

    scores = score_pairs(claims, oracle_rankings, docs_by_id, tokenizer, model, label_idx, device)
    print("\n=== Oracle-evidence diagnostic (not a valid submission score) ===")
    oracle_fixed = predictions_at_threshold(scores, THRESHOLD)
    metrics(f"Oracle docs, threshold={THRESHOLD:.3f}", oracle_fixed, gold)

    best = None
    for threshold in THRESHOLDS:
        pred = predictions_at_threshold(scores, threshold)
        f1, tp, fp, fn = triple_f1(pred, gold)
        candidate = (f1, threshold, tp, fp, fn, pred)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    f1, threshold, tp, fp, fn, pred = best
    metrics(f"Oracle docs, best validation threshold={threshold:.3f}", pred, gold)
    print("\nInterpretation:")
    print("- A large oracle-vs-actual F1 gap suggests retrieval is a major bottleneck.")
    print("- If oracle F1 is still low, verifier labels/thresholding are also a major bottleneck.")
    print("- Oracle results use gold evidence documents and must not be reported as real system performance.")


if __name__ == "__main__":
    main()

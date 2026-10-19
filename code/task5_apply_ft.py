#!/usr/bin/env python3
"""Turn artifacts/ft_scores.json (from task5_finetune_colab.py) into the cached
verification CSVs. Top-2 reranked docs per claim: rank 1 accepted at margin >= t1,
rank 2 added only at margin >= t2. Both thresholds chosen on validation only."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART, DATA = ROOT / "artifacts", ROOT / "data"
N = 2

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

scores = json.load(open(ART / "ft_scores.json"))
val = read_jsonl(DATA / "val_claims.jsonl")
gold = set()
for c in val:
    for e in c.get("evidence", []) or []:
        if str(e.get("label", "")).upper() in ("SUPPORT", "CONTRADICT"):
            gold.add((int(c["id"]), int(e["doc_id"]), str(e["label"]).upper()))

def parse(sc):
    rows = []
    for k, (ps, pc, pn, _) in sc.items():
        cid, did, pos = map(int, k.split("|"))
        if pos < N:
            rows.append((cid, did, pos, max(ps, pc) - pn, "SUPPORT" if ps >= pc else "CONTRADICT"))
    return rows

def predict(rows, t1, t2):
    return {(cid, did, lab) for cid, did, pos, m, lab in rows if m >= (t1 if pos == 0 else t2)}

val_rows, eval_rows = parse(scores["val"]), parse(scores["eval"])
best = (0, 0, 0)
for t1 in np.arange(0.0, 1.0, 0.05):
    for t2 in np.arange(0.0, 1.0, 0.05):
        pred = predict(val_rows, t1, t2)
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]:
            best = (f1, t1, t2)
f1, t1, t2 = best
print(f"N={N} t1={t1:.2f} t2={t2:.2f} validation triple-F1={f1:.4f}")

def write(path, triples):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "doc_id", "label"])
        for t in sorted(triples):
            w.writerow(t)

val_pred, eval_pred = predict(val_rows, t1, t2), predict(eval_rows, t1, t2)
write(ART / "val_verification.csv", val_pred)
write(ART / "eval_verification.csv", eval_pred)
print(f"val predictions: {len(val_pred)}, eval predictions: {len(eval_pred)}")

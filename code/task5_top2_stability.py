#!/usr/bin/env python3
"""Task 5: how stable is validation F1 as the rank-2 / rank-1 thresholds move (N=2)."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sc = json.load(open(ROOT / "artifacts" / "ft_scores_no_nei.json"))["val"]
claims = [json.loads(l) for l in open(ROOT / "data" / "val_claims.jsonl") if l.strip()]
gold = set()
for c in claims:
    for e in c.get("evidence", []) or []:
        lab = str(e.get("label", "")).upper()
        if lab in ("SUPPORT", "CONTRADICT"):
            gold.add((int(c["id"]), int(e["doc_id"]), lab))
rows = defaultdict(list)
for k, (ps, pc, pn, _) in sc.items():
    cid, did, pos = map(int, k.split("|"))
    rows[cid].append((pos, did, max(ps, pc) - pn, "SUPPORT" if ps >= pc else "CONTRADICT"))

def f1(t1, t2, N=2):
    pred = set()
    for cid, rs in rows.items():
        for pos, did, m, lab in rs:
            if pos < N and m >= (t1 if pos == 0 else t2):
                pred.add((cid, did, lab))
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    return 2 * tp / max(1, 2 * tp + fp + fn)

print("t1=0.70, N=2, F1 by rank-2 threshold:")
for t2 in (0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99, 1.01):
    print(f"  t2={t2:.2f}  F1={f1(0.70, t2):.4f}")
print("t2=0.95, F1 by rank-1 threshold:")
for t1 in (0.60, 0.65, 0.70, 0.75, 0.80):
    print(f"  t1={t1:.2f}  F1={f1(t1, 0.95):.4f}")

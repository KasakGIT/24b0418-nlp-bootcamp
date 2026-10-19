#!/usr/bin/env python3
"""Task 5: sweep rank-1 and rank-2 margin thresholds on validation (top-2 docs per claim)."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sc = json.load(open(ROOT / "artifacts" / "ft_scores_no_nei.json"))["val"]
claims = [json.loads(l) for l in open(ROOT / "data" / "val_claims.jsonl") if l.strip()]
gold = set(); ndocs = defaultdict(set)
for c in claims:
    for e in c.get("evidence", []) or []:
        lab = str(e.get("label", "")).upper()
        if lab in ("SUPPORT", "CONTRADICT"):
            gold.add((int(c["id"]), int(e["doc_id"]), lab))
            ndocs[int(c["id"])].add(int(e["doc_id"]))
print("claims with gold:", len(ndocs), "| with 2+ gold docs:", sum(len(v) > 1 for v in ndocs.values()))
rows = defaultdict(list)
for k, (ps, pc, pn, _) in sc.items():
    cid, did, pos = map(int, k.split("|"))
    rows[cid].append((pos, did, max(ps, pc) - pn, "SUPPORT" if ps >= pc else "CONTRADICT"))
best = (0,)
for N in (1, 2, 3):
    for t1 in np.arange(0, 1.0, 0.05):
        for t2 in np.arange(0, 1.0, 0.05):
            pred = set()
            for cid, rs in rows.items():
                for pos, did, m, lab in rs:
                    if pos < N and m >= (t1 if pos == 0 else t2):
                        pred.add((cid, did, lab))
            tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            if f1 > best[0]:
                best = (f1, N, round(float(t1), 2), round(float(t2), 2), tp, fp, fn)
print("best (F1, N, thr_rank1, thr_other, TP, FP, FN):", best)

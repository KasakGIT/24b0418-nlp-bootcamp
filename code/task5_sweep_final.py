import sys, json, csv
import numpy as np
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from task5_verify_finalize import read_jsonl, gold_triples, triple_f1

raw = json.load(open(ROOT / "artifacts/val_rerank_scores.json"))
names = list(raw)
print("models:", names)

rows = defaultdict(list)
for key in raw[names[0]]:
    cid, did, pos = map(int, key.split("|"))
    v = [raw[n][key] for n in names if key in raw[n]]
    ps, pc, pn = (float(np.mean([x[i] for x in v])) for i in range(3))
    rows[cid].append((pos, did, ps, pc, pn))
for cid in rows:
    rows[cid].sort()

gold = set()
for c in read_jsonl(ROOT / "data/val_claims.jsonl"):
    gold |= gold_triples(c)

def predict(N, thr, gap):
    out = set()
    for cid, rs in rows.items():
        c = sorted(((max(ps, pc) - pn, did, "SUPPORT" if ps >= pc else "CONTRADICT")
                    for pos, did, ps, pc, pn in rs[:N]), reverse=True)
        if not c or c[0][0] < thr:
            continue
        if len(c) > 1 and c[0][0] - c[1][0] < gap:
            continue
        out.add((cid, c[0][1], c[0][2]))
    return out

results = []
for N in (1, 2, 3, 4, 5, 10):
    for gap in (0.0, 0.05, 0.1, 0.2):
        for thr in np.arange(-0.5, 0.951, 0.05):
            f1, tp, fp, fn = triple_f1(predict(N, float(thr), gap), gold)
            results.append((f1, N, gap, round(float(thr), 2), tp, fp, fn))

print("\nbest per N (gap=0 only):")
for N in (1, 2, 3, 4, 5, 10):
    r = max((x for x in results if x[1] == N and x[2] == 0.0), key=lambda x: x[0])
    print(f"N={N:<2d} thr={r[3]:+.2f} F1={r[0]:.4f} TP={r[4]} FP={r[5]} FN={r[6]}")

print("\ntop 8 overall (with abstain gap):")
for r in sorted(results, reverse=True)[:8]:
    print(f"N={r[1]:<2d} gap={r[2]:.2f} thr={r[3]:+.2f} F1={r[0]:.4f} TP={r[4]} FP={r[5]} FN={r[6]}")

f1, N, gap, thr, *_ = max(results)
pred = predict(N, thr, gap)
with open(ROOT / "artifacts/val_verification_best.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["claim_id", "doc_id", "label"])
    for t in sorted(pred): w.writerow(t)
print(f"\nwrote val_verification_best.csv ({len(pred)} rows) for N={N} gap={gap} thr={thr}")
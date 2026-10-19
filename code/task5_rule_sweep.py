import json, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from task5_verify_finalize import gold_triples, read_jsonl, triple_f1

claims = read_jsonl(ROOT / "data" / "val_claims.jsonl")
gold = set()
for c in claims:
    gold.update(gold_triples(c))

raw = json.loads((ROOT / "artifacts/task5_val_pair_scores.json").read_text())
scores = {int(k): v for k, v in raw.items()}   # cid -> list of rows

def predict(threshold, margin):
    """margin=None -> current baseline (no NEUTRAL check)."""
    out = set()
    for cid, rows in scores.items():
        elig = []
        for r in rows:
            conf = max(r["support_prob"], r["contradict_prob"])
            if conf < threshold:
                continue
            if margin is not None and conf - r["neutral_prob"] < margin:
                continue
            elig.append((conf, r))
        elig.sort(key=lambda x: x[0], reverse=True)
        seen, n = set(), 0
        for conf, r in elig:
            key = (r["doc_id"], r["label"])
            if key in seen:
                continue
            seen.add(key)
            out.add((cid, int(r["doc_id"]), r["label"]))
            n += 1
            if n >= 3:
                break
    return out

rules = {"baseline": None, "beat-neutral (0)": 0.0,
         "margin 0.05": 0.05, "margin 0.10": 0.10}
thresholds = [round(0.05 + 0.025 * i, 3) for i in range(35)]

print(f"gold triples: {len(gold)}")
for name, m in rules.items():
    best = max(((triple_f1(predict(t, m), gold), t) for t in thresholds),
               key=lambda x: x[0][0])
    (f1, tp, fp, fn), t = best
    print(f"{name:18s} best_thr={t:.3f}  F1={f1:.4f}  TP={tp} FP={fp} FN={fn}")
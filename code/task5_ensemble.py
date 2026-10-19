import sys, json
import numpy as np
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from task5_window_scores import predict, gold_triples, read_jsonl, triple_f1

A = json.loads((ROOT / "artifacts/task5_val_window_scores_w2.json").read_text())["margin"]
B = json.loads((ROOT / "artifacts/val_scores_DeBERTa-v3-base-mnli-fever-anli_w2.json").read_text())["margin"]


def blend(A, B):
    out = {}
    for cid, rows in A.items():
        bmap = {r["doc_id"]: r for r in B.get(cid, [])}
        new = []
        for r in rows:                      # keeps MiniLM's retrieval-rank order
            b = bmap.get(r["doc_id"])
            if b is None:
                continue
            ps = (r["support_prob"] + b["support_prob"]) / 2
            pc = (r["contradict_prob"] + b["contradict_prob"]) / 2
            pn = (r["neutral_prob"] + b["neutral_prob"]) / 2
            conf = max(ps, pc)
            new.append({"doc_id": r["doc_id"], "label": "SUPPORT" if ps >= pc else "CONTRADICT",
                        "support_prob": ps, "contradict_prob": pc, "neutral_prob": pn,
                        "confidence": conf, "margin": conf - pn})
        out[cid] = new
    return out


def with_bias(S, bias):
    """Handicap CONTRADICT: choose it only if pc - bias > ps."""
    out = {}
    for cid, rows in S.items():
        new = []
        for r in rows:
            ps, pc, pn = r["support_prob"], r["contradict_prob"], r["neutral_prob"]
            if ps >= pc - bias:
                lab, p = "SUPPORT", ps
            else:
                lab, p = "CONTRADICT", pc
            new.append({**r, "label": lab, "confidence": p, "margin": p - pn})
        out[cid] = new
    return out


gold = set()
for c in read_jsonl(ROOT / "data/val_claims.jsonl"):
    gold |= gold_triples(c)

# ---------- 1) model / rank-gate comparison ----------
sets = {"MiniLM": A, "Moritz": B, "Blend": blend(A, B)}
grids = {"margin": np.arange(-0.5, 0.951, 0.05), "confidence": np.arange(0.05, 0.951, 0.025)}
for name, S in sets.items():
    for N in (3, 5, 10):
        scores = {cid: rows[:N] for cid, rows in S.items()}
        for k in (1, 2):
            for key, grid in grids.items():
                (f1, tp, fp, fn), t = max(
                    ((triple_f1(predict(scores, key, float(t), k), gold), round(float(t), 3)) for t in grid),
                    key=lambda x: x[0][0])
                print(f"{name:7s} top{N:<2d}docs top_k={k} {key:10s} thr={t:.3f} F1={f1:.4f} TP={tp} FP={fp} FN={fn}")

# ---------- 2) plateau curve ----------
print("\nBlend top3 top_k=1 margin curve")
S3 = {cid: rows[:3] for cid, rows in blend(A, B).items()}
for t in np.arange(-0.30, 0.31, 0.05):
    f1, tp, fp, fn = triple_f1(predict(S3, "margin", float(t), 1), gold)
    print(f"thr={t:+.2f} F1={f1:.4f} TP={tp} FP={fp}")

# ---------- 3) label breakdown ----------
print("\n--- label breakdown, blend top3 top_k=1 thr=-0.05 ---")
pred = predict(S3, "margin", -0.05, 1)
print("gold labels:", Counter(l for _, _, l in gold))
print("pred labels:", Counter(l for _, _, l in pred))
for lab in ("SUPPORT", "CONTRADICT"):
    p = {x for x in pred if x[2] == lab}
    g = {x for x in gold if x[2] == lab}
    print(f"{lab:10s} predicted={len(p)} TP={len(p & g)} "
          f"precision={len(p & g)/max(1,len(p)):.2f} recall={len(p & g)/max(1,len(g)):.2f}")
gd = {(c, d): l for c, d, l in gold}
wrong = sum(1 for c, d, l in pred if (c, d) in gd and gd[(c, d)] != l)
print("right doc but wrong label:", wrong)

# ---------- 4) CONTRADICT bias sweep ----------
print("\n--- CONTRADICT handicap sweep (blend, top3, top_k=1) ---")
for bias in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
    Sb = with_bias(S3, bias)
    best = max(((triple_f1(predict(Sb, "margin", float(t), 1), gold), round(float(t), 3))
                for t in np.arange(-0.5, 0.51, 0.05)), key=lambda x: x[0][0])
    (f1, tp, fp, fn), t = best
    pb = predict(Sb, "margin", t, 1)
    n_c = sum(1 for x in pb if x[2] == "CONTRADICT")
    print(f"bias={bias:.2f} thr={t:+.2f} F1={f1:.4f} TP={tp} FP={fp} "
          f"preds={len(pb)} contradict_share={n_c/max(1,len(pb)):.2f}")
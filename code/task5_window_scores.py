#!/usr/bin/env python3
"""Sentence-window NLI scoring on validation claims (diagnostic only).
Run from project root: python code/task5_window_scores.py [max_window]
"""
import json, re, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from task5_verify_finalize import (
    read_jsonl, gold_triples, read_trec, find_label_indices,
    triple_f1, MODEL_NAME, TOP_DOCS,
)

MAX_WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 2   # windows of 1..MAX_WIN sentences
BATCH = 64
MAXLEN = 128
CACHE = ROOT / "artifacts" / f"task5_val_window_scores_w{MAX_WIN}.json"


def sentences(d):
    ab = d.get("abstract", [])
    if isinstance(ab, str):
        ab = re.split(r"(?<=[.!?])\s+", ab)
    return [str(s).strip() for s in ab if str(s).strip()]


def windows(sents):
    out = []
    for n in range(1, MAX_WIN + 1):
        for i in range(len(sents) - n + 1):
            out.append(" ".join(sents[i:i + n]))
    return out or [" ".join(sents)]


def build_cache():
    corpus = {int(d["doc_id"]): d for d in read_jsonl(ROOT / "data/corpus.jsonl")}
    claims = read_jsonl(ROOT / "data/val_claims.jsonl")
    _raw = json.loads((ROOT / "artifacts/task5_val_pair_scores.json").read_text())
    rankings = {int(k): [int(r["doc_id"]) for r in v] for k, v in _raw.items()}
    print(f"val claims with rankings: {len(rankings)}, "
          f"docs in corpus: {sum(int(d) in corpus for v in rankings.values() for d in v)}")

    pairs, owners = [], []
    for c in claims:
        cid = int(c["id"])
        for did in rankings.get(cid, [])[:TOP_DOCS]:
            d = corpus.get(int(did))
            if d is None:
                continue
            for w in windows(sentences(d)):
                pairs.append((w, str(c["claim"])))
                owners.append((cid, int(did)))
    print(f"{len(pairs)} window pairs")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
    idx = find_label_indices(model.config)

    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))  # speed
    probs_all = [None] * len(pairs)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), BATCH):
            ids = order[s:s + BATCH]
            enc = tok([pairs[i][0] for i in ids], [pairs[i][1] for i in ids],
                      padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            p = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            for i, row in zip(ids, p):
                probs_all[i] = row
            if (s // BATCH) % 50 == 0:
                print(f"{s}/{len(order)}  {time.time()-t0:.0f}s", flush=True)

    # per (claim, doc): keep best window under two selection modes
    best = {}
    for (cid, did), p in zip(owners, probs_all):
        ps, pc, pn = (float(p[idx["SUPPORT"]]), float(p[idx["CONTRADICT"]]), float(p[idx["NEUTRAL"]]))
        conf = max(ps, pc)
        row = {"doc_id": did, "label": "SUPPORT" if ps >= pc else "CONTRADICT",
               "support_prob": ps, "contradict_prob": pc, "neutral_prob": pn,
               "confidence": conf, "margin": conf - pn}
        cur = best.setdefault((cid, did), {"conf": row, "margin": row})
        if row["confidence"] > cur["conf"]["confidence"]:
            cur["conf"] = row
        if row["margin"] > cur["margin"]["margin"]:
            cur["margin"] = row
    out = {"conf": {}, "margin": {}}
    for (cid, did), v in best.items():
        for mode in out:
            out[mode].setdefault(str(cid), []).append(v[mode])
    CACHE.write_text(json.dumps(out))
    return out


def predict(scores, key, thr, top_k=3):
    res = set()
    for cid, rows in scores.items():
        el = sorted((r for r in rows if r[key] >= thr), key=lambda r: r[key], reverse=True)
        seen, n = set(), 0
        for r in el:
            k = (r["doc_id"], r["label"])
            if k in seen:
                continue
            seen.add(k)
            res.add((int(cid), int(r["doc_id"]), r["label"]))
            n += 1
            if n >= top_k:
                break
    return res


def main():
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else build_cache()
    claims = read_jsonl(ROOT / "data/val_claims.jsonl")
    gold = set()
    for c in claims:
        gold.update(gold_triples(c))
    print(f"gold triples: {len(gold)}")

    thrs = [round(x, 3) for x in np.arange(0.05, 0.951, 0.025)]
    for mode, key in [("conf", "confidence"), ("margin", "margin")]:
        scores = cache[mode]
        for thr_key in ([key] if key == "confidence" else ["margin"]):
            lo = thrs if key == "confidence" else [round(x, 3) for x in np.arange(-0.5, 0.901, 0.05)]
            best = max(((triple_f1(predict(scores, thr_key, t), gold), t) for t in lo),
                       key=lambda x: x[0][0])
            (f1, tp, fp, fn), t = best
            print(f"window select={mode:6s} thr_on={thr_key:10s} best_thr={t:.3f} "
                  f"F1={f1:.4f} TP={tp} FP={fp} FN={fn}")

    for k in (1, 2, 3):
        for mode, key in [("conf", "confidence"), ("margin", "margin")]:
            scores = cache[mode]
            lo = thrs if key == "confidence" else [round(x, 3) for x in np.arange(-0.5, 0.901, 0.05)]
            (f1, tp, fp, fn), t = max(((triple_f1(predict(scores, key, t, k), gold), t) for t in lo),
                                      key=lambda x: x[0][0])
            print(f"top_k={k} {mode:6s} thr={t:.3f} F1={f1:.4f} TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
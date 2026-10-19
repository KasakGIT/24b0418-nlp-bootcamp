#!/usr/bin/env python3
"""Validation ranking with the exact Task 3 round-2 winner:
union_depth=100, rerank_depth=10, rrf_k=15. Writes predictions/t3_rerank_val.trec only."""
import json, time
from pathlib import Path
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA, PRED = ROOT / "data", ROOT / "predictions"
GENERAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
UNION_DEPTH, RERANK_DEPTH, RRF_K = 100, 10, 15


def load_jsonl(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

def text_of_doc(d):
    ab = d.get("abstract", [])
    if isinstance(ab, list): ab = " ".join(ab)
    return (d.get("title", "") + " " + ab).strip()

def text_of_claim(c): return c.get("claim", "").strip()

def gold_ids(c):
    out = set()
    for x in c.get("evidence", []) or []:
        if isinstance(x, dict):
            for key in ("abstract_id", "doc_id", "id"):
                if key in x: out.add(str(x[key])); break
    return out

def ndcg10(ranked, gold):
    if not gold: return None
    rel = np.array([1.0 if d in gold else 0.0 for d in ranked[:10]])
    disc = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    ideal = float(np.sum(disc[:min(len(gold), 10)]))
    return float(np.sum(rel * disc)) / ideal if ideal > 0 else 0.0

def rrf_fuse(b, d, k):
    s = {}
    for r, x in enumerate(b, 1): s[x] = s.get(x, 0.0) + 1.0 / (k + r)
    for r, x in enumerate(d, 1): s[x] = s.get(x, 0.0) + 1.0 / (k + r)
    return [x for x, _ in sorted(s.items(), key=lambda t: (-t[1], t[0]))]

def union_candidates(b, d, depth):
    seen, out = set(), []
    for rk in (b[:depth], d[:depth]):
        for x in rk:
            if x not in seen: seen.add(x); out.append(x)
    return out

def rerank(q, ids, docs_by_id, ce):
    sc = ce.predict([(q, text_of_doc(docs_by_id[d])) for d in ids], batch_size=32, show_progress_bar=False)
    order = sorted(range(len(ids)), key=lambda i: (-float(sc[i]), ids[i]))
    return [ids[i] for i in order]


corpus = load_jsonl(DATA / "corpus.jsonl")
val = load_jsonl(DATA / "val_claims.jsonl")
docs_by_id = {str(d["doc_id"]): d for d in corpus}
doc_ids = [str(d["doc_id"]) for d in corpus]
doc_texts = [text_of_doc(d) for d in corpus]

print("BM25...")
bm25 = BM25Okapi([t.lower().split() for t in doc_texts], k1=2.0, b=0.3)
print("Encoding corpus (a few minutes)...")
enc = SentenceTransformer(GENERAL_MODEL)
emb = enc.encode(doc_texts, batch_size=64, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=True)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(emb.astype("float32"))
ce = CrossEncoder(RERANK_MODEL)

rankings, t0 = {}, time.time()
for n, c in enumerate(val, 1):
    cid, q = str(c["id"]), text_of_claim(c)
    bs = bm25.get_scores(q.lower().split())
    b_rank = [doc_ids[i] for i in np.argsort(-bs, kind="stable")[:100]]
    qe = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    _, inds = index.search(qe, 100)
    d_rank = [doc_ids[i] for i in inds[0]]
    pool = union_candidates(b_rank, d_rank, UNION_DEPTH)
    top = rerank(q, pool, docs_by_id, ce)[:RERANK_DEPTH]
    seen = set(top)
    rrf = rrf_fuse(b_rank, d_rank, RRF_K)[:100]
    rankings[cid] = (top + [d for d in rrf if d not in seen])[:100]
    if n % 10 == 0:
        print(f"{n}/{len(val)} claims, {time.time()-t0:.0f}s", flush=True)

vals = [ndcg10(rankings[str(c["id"])], gold_ids(c)) for c in val]
print(f"validation nDCG@10 = {np.mean([v for v in vals if v is not None]):.4f}  (expect ~0.8676)")

with open(PRED / "t3_rerank_val.trec", "w", encoding="utf-8") as f:
    for cid, rk in rankings.items():
        for r, d in enumerate(rk, 1):
            f.write(f"{cid} Q0 {d} {r} {1.0/r:.8f} t3_rerank_val\n")
print("wrote predictions/t3_rerank_val.trec")
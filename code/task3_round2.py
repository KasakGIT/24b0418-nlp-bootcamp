#!/usr/bin/env python3
"""
Task 3 Round 2
Targeted improvements to hybrid retrieval + cross-encoder reranking.

Selection is ONLY by validation nDCG@10.
Eval is touched only after the best configuration is selected.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRED = ROOT / "predictions"
ART = ROOT / "artifacts"
PRED.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)

CORPUS_FILE = DATA / "corpus.jsonl"
VAL_FILE = DATA / "val_claims.jsonl"
EVAL_FILE = DATA / "eval_claims.jsonl"

GENERAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Current Task 3 winner:
BASELINE = 0.8645

# Round-2 search, concentrated around the previous winners.
RRF_K_VALUES = [5, 10, 15, 20, 30]
RERANK_DEPTHS = [10, 15, 20, 30, 40]

# Instead of reranking only the RRF top-k, first create a larger
# candidate pool from the BM25 and dense rankings.
UNION_DEPTHS = [50, 100, 200]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def text_of_doc(d):
    abstract = d.get("abstract", [])
    if isinstance(abstract, list):
        abstract = " ".join(abstract)
    return (d.get("title", "") + " " + abstract).strip()


def text_of_claim(c):
    return c.get("claim", "").strip()


def gold_ids(c):
    ev = c.get("evidence", [])
    if isinstance(ev, dict):
        ev = ev.get("abstracts", [])
    out = set()
    for x in ev:
        if isinstance(x, str):
            out.add(x)
        elif isinstance(x, dict):
            for key in ("abstract_id", "doc_id", "id"):
                if key in x:
                    out.add(str(x[key]))
                    break
    return out


def ndcg10(ranked, gold):
    if not gold:
        return None
    rel = np.array([1.0 if d in gold else 0.0 for d in ranked[:10]])
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float(np.sum(rel * discounts))
    ideal_n = min(len(gold), 10)
    ideal = float(np.sum(discounts[:ideal_n]))
    return dcg / ideal if ideal > 0 else 0.0


def mean_ndcg(rankings, claims):
    vals = []
    for c in claims:
        score = ndcg10(rankings[str(c["id"])], gold_ids(c))
        if score is not None:
            vals.append(score)
    return float(np.mean(vals))


def top_rank_dict(ranking):
    return {doc_id: rank for rank, doc_id in enumerate(ranking, start=1)}


def rrf_fuse(bm25_rank, dense_rank, k):
    scores = {}
    for rank, doc_id in enumerate(bm25_rank, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(dense_rank, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return [
        d for d, _ in sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0])
        )
    ]


def union_candidates(bm25_rank, dense_rank, depth):
    seen = set()
    out = []
    for ranking in (bm25_rank[:depth], dense_rank[:depth]):
        for d in ranking:
            if d not in seen:
                seen.add(d)
                out.append(d)
    return out


def rerank(query, candidate_ids, docs_by_id, cross_encoder):
    pairs = [(query, text_of_doc(docs_by_id[d])) for d in candidate_ids]
    scores = cross_encoder.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False
    )
    order = sorted(
        range(len(candidate_ids)),
        key=lambda i: (-float(scores[i]), candidate_ids[i])
    )
    return [candidate_ids[i] for i in order]


def make_trec(path, rankings, run_name):
    with open(path, "w", encoding="utf-8") as f:
        for claim_id in rankings:
            ranking = rankings[claim_id][:100]
            for rank, doc_id in enumerate(ranking, start=1):
                score = 1.0 / rank
                f.write(
                    f"{claim_id} Q0 {doc_id} {rank} {score:.8f} {run_name}\n"
                )


print("Task 3 Round 2 starting...")
print("Project root:", ROOT)

corpus = load_jsonl(CORPUS_FILE)
val = load_jsonl(VAL_FILE)
eval_claims = load_jsonl(EVAL_FILE)

docs_by_id = {str(d["doc_id"]): d for d in corpus}
doc_ids = [str(d["doc_id"]) for d in corpus]
doc_texts = [text_of_doc(d) for d in corpus]

# ---------- BM25 ----------
print("Building BM25...")
tokenized = [x.lower().split() for x in doc_texts]
bm25 = BM25Okapi(tokenized, k1=2.0, b=0.3)

# ---------- Dense ----------
print("Loading MiniLM...")
encoder = SentenceTransformer(GENERAL_MODEL)

print("Encoding corpus...")
t0 = time.perf_counter()
doc_emb = encoder.encode(
    doc_texts,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)
print(f"Corpus encoding time: {time.perf_counter() - t0:.2f}s")

index = faiss.IndexFlatIP(doc_emb.shape[1])
index.add(doc_emb.astype("float32"))

def get_base_rankings(claims):
    bm = {}
    dense = {}

    for c in claims:
        cid = str(c["id"])
        q = text_of_claim(c)

        b_scores = bm25.get_scores(q.lower().split())
        b_order = np.argsort(-b_scores, kind="stable")
        bm[cid] = [doc_ids[i] for i in b_order[:100]]

        q_emb = encoder.encode(
            [q],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        _, inds = index.search(q_emb, 100)
        dense[cid] = [doc_ids[i] for i in inds[0]]

    return bm, dense


print("Computing validation rankings...")
val_bm25, val_dense = get_base_rankings(val)

# ---------- RRF search ----------
print("\nSearching finer RRF constants...")
rrf_results = {}

for k in RRF_K_VALUES:
    rankings = {}
    for c in val:
        cid = str(c["id"])
        rankings[cid] = rrf_fuse(val_bm25[cid], val_dense[cid], k)[:100]

    score = mean_ndcg(rankings, val)
    rrf_results[k] = score
    print(f"RRF k={k}: validation nDCG@10={score:.4f}")

best_rrf_k = max(rrf_results, key=rrf_results.get)
best_rrf_score = rrf_results[best_rrf_k]

print(
    f"BEST RRF: k={best_rrf_k}, "
    f"validation nDCG@10={best_rrf_score:.4f}"
)

# ---------- Cross encoder ----------
print("\nLoading cross-encoder...")
cross_encoder = CrossEncoder(RERANK_MODEL)

# We test two candidate-pool strategies:
# A) RRF pool at a chosen depth
# B) union of BM25 + dense candidates.
#
# First, use the larger union pools. This can recover documents
# that RRF ranking pushes below the reranking cutoff.
best_config = None
best_score = -1.0
all_results = []

for union_depth in UNION_DEPTHS:
    print(f"\n=== Candidate union depth: {union_depth} ===")

    # Build candidate unions once.
    candidates = {}
    for c in val:
        cid = str(c["id"])
        candidates[cid] = union_candidates(
            val_bm25[cid],
            val_dense[cid],
            union_depth
        )

    for rerank_depth in RERANK_DEPTHS:
        rankings = {}
        start = time.perf_counter()

        for c in val:
            cid = str(c["id"])
            q = text_of_claim(c)

            pool = candidates[cid]
            reranked = rerank(
                q,
                pool,
                docs_by_id,
                cross_encoder
            )

            # Keep reranked top-N, then fill remaining positions
            # from the original RRF ranking so the run has 100 docs.
            rrf_rank = rrf_fuse(
                val_bm25[cid],
                val_dense[cid],
                best_rrf_k
            )[:100]

            top = reranked[:rerank_depth]
            seen = set(top)
            final = top + [d for d in rrf_rank if d not in seen]

            rankings[cid] = final[:100]

        elapsed = time.perf_counter() - start
        latency_ms = elapsed / max(len(val), 1) * 1000.0
        score = mean_ndcg(rankings, val)

        all_results.append({
            "candidate_pool": f"union_{union_depth}",
            "union_depth": union_depth,
            "rerank_depth": rerank_depth,
            "rrf_k": best_rrf_k,
            "ndcg10": score,
            "latency_ms_per_query": latency_ms,
        })

        print(
            f"union={union_depth:3d}, "
            f"rerank={rerank_depth:2d}: "
            f"nDCG@10={score:.4f}, "
            f"latency={latency_ms:.1f} ms/query"
        )

        if score > best_score:
            best_score = score
            best_config = all_results[-1]

print("\n==============================")
print("TASK 3 ROUND 2 BEST")
print("==============================")
print(json.dumps(best_config, indent=2))

summary = {
    "baseline_task3_rerank_ndcg10": BASELINE,
    "best_rrf_k": best_rrf_k,
    "best_rrf_ndcg10": best_rrf_score,
    "best_round2": best_config,
    "all_round2_results": all_results,
}

with open(ART / "task3_round2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

# ---------- Only proceed to eval if we beat the existing winner ----------
if best_score <= BASELINE:
    print(
        f"\nRound 2 did NOT beat baseline {BASELINE:.4f}. "
        f"Keeping existing t3_rerank.trec."
    )
    print("Task 3 Round 2 complete.")
    raise SystemExit(0)

print(
    f"\nNEW BEST! {best_score:.4f} > baseline {BASELINE:.4f}"
)

# Recompute eval rankings using selected settings.
print("Computing evaluation rankings...")
eval_bm25, eval_dense = get_base_rankings(eval_claims)

eval_rankings = {}

for c in eval_claims:
    cid = str(c["id"])
    q = text_of_claim(c)

    pool = union_candidates(
        eval_bm25[cid],
        eval_dense[cid],
        best_config["union_depth"]
    )

    reranked = rerank(
        q,
        pool,
        docs_by_id,
        cross_encoder
    )

    rrf_rank = rrf_fuse(
        eval_bm25[cid],
        eval_dense[cid],
        best_rrf_k
    )[:100]

    top = reranked[:best_config["rerank_depth"]]
    seen = set(top)
    final = top + [d for d in rrf_rank if d not in seen]

    eval_rankings[cid] = final[:100]

# Keep required filename.
make_trec(
    PRED / "t3_rerank.trec",
    eval_rankings,
    "t3_rerank_round2"
)

# Also save an explicit round-2 file.
make_trec(
    PRED / "t3_rerank_round2.trec",
    eval_rankings,
    "t3_rerank_round2"
)

print("Updated:")
print("  predictions/t3_rerank.trec")
print("  predictions/t3_rerank_round2.trec")
print("  artifacts/task3_round2_summary.json")
print("Task 3 Round 2 COMPLETE.")

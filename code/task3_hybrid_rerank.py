import json
import os
import time
import gc
import numpy as np
import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
PRED_DIR = "predictions"
ART_DIR = "artifacts"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

SEED = 42

BM25_K1 = 2.0
BM25_B = 0.3

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

RRF_K_VALUES = [20, 40, 60, 100]
RERANK_K_VALUES = [20, 50, 100]


# ============================================================
# DATA
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


corpus = load_jsonl(f"{DATA_DIR}/corpus.jsonl")
train = load_jsonl(f"{DATA_DIR}/train_claims.jsonl")
val = load_jsonl(f"{DATA_DIR}/val_claims.jsonl")
eval_claims = load_jsonl(f"{DATA_DIR}/eval_claims.jsonl")

doc_ids = [int(x["doc_id"]) for x in corpus]
doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

def doc_text(d):
    abstract = d["abstract"]
    if isinstance(abstract, list):
        abstract = " ".join(abstract)
    return f'{d["title"]} {abstract}'


documents = [doc_text(d) for d in corpus]


# ============================================================
# METRICS
# ============================================================

def gold_docs(claim):
    return {
        int(e["doc_id"])
        for e in claim.get("evidence", [])
    }


def ndcg10(ranked_docs, gold):
    if not gold:
        return 0.0

    dcg = 0.0

    for rank, doc_id in enumerate(ranked_docs[:10], start=1):
        if doc_id in gold:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_n = min(len(gold), 10)

    idcg = sum(
        1.0 / np.log2(rank + 1)
        for rank in range(1, ideal_n + 1)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(ranked_lists, claims):
    vals = []

    for ranked, claim in zip(ranked_lists, claims):
        gold = gold_docs(claim)

        # Retrieval metric is meaningful for claims with evidence.
        if gold:
            vals.append(ndcg10(ranked, gold))

    return float(np.mean(vals)) if vals else 0.0


# ============================================================
# TREC WRITER
# ============================================================

def write_trec(path, claims, rankings, run_name):
    with open(path, "w", encoding="utf-8") as f:
        for claim, ranking in zip(claims, rankings):

            claim_id = int(claim["id"])

            # Guarantee exactly 100 unique documents.
            ranking = list(dict.fromkeys(ranking))[:100]

            if len(ranking) < 100:
                raise RuntimeError(
                    f"Claim {claim_id} has only {len(ranking)} unique docs"
                )

            for rank, doc_id in enumerate(ranking, start=1):
                # Rank-based score guarantees monotonically
                # decreasing scores and does not affect ranking.
                score = 1.0 / rank

                f.write(
                    f"{claim_id} Q0 {int(doc_id)} "
                    f"{rank} {score:.8f} {run_name}\n"
                )


# ============================================================
# BM25
# ============================================================

print("=" * 70)
print("TASK 3 — HYBRID + CROSS-ENCODER RERANKING")
print("=" * 70)

print("\nBuilding BM25...")

tokenized_docs = [
    text.lower().split()
    for text in documents
]

bm25 = BM25Okapi(
    tokenized_docs,
    k1=BM25_K1,
    b=BM25_B
)


def bm25_retrieve(claims):
    rankings = []

    for claim in claims:
        query = claim["claim"].lower().split()
        scores = bm25.get_scores(query)

        # Full ranking, then top 100.
        idx = np.argsort(scores)[::-1][:100]

        rankings.append(
            [doc_ids[i] for i in idx]
        )

    return rankings


print("Retrieving BM25...")
val_bm25 = bm25_retrieve(val)
eval_bm25 = bm25_retrieve(eval_claims)

print(
    f"BM25 validation nDCG@10: "
    f"{evaluate(val_bm25, val):.4f}"
)


# ============================================================
# MINI-LM DENSE RETRIEVAL
# ============================================================

print("\nLoading MiniLM...")
dense_model = SentenceTransformer(DENSE_MODEL)

print("Encoding corpus...")

t0 = time.perf_counter()

doc_embeddings = dense_model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

encoding_time = time.perf_counter() - t0

doc_embeddings = np.asarray(
    doc_embeddings,
    dtype=np.float32
)

print(
    f"Corpus encoding time: "
    f"{encoding_time:.2f} sec"
)

print("Saving document embeddings for Task 4...")
np.save(
    f"{ART_DIR}/minilm_doc_embeddings.npy",
    doc_embeddings
)

np.save(
    f"{ART_DIR}/minilm_doc_ids.npy",
    np.asarray(doc_ids, dtype=np.int64)
)

index = faiss.IndexFlatIP(
    doc_embeddings.shape[1]
)

index.add(doc_embeddings)


def dense_retrieve(claims, save_query_embeddings=False, prefix=""):
    texts = [c["claim"] for c in claims]

    q_embeddings = dense_model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    q_embeddings = np.asarray(
        q_embeddings,
        dtype=np.float32
    )

    if save_query_embeddings:
        np.save(
            f"{ART_DIR}/{prefix}_query_embeddings.npy",
            q_embeddings
        )

    scores, indices = index.search(
        q_embeddings,
        100
    )

    rankings = []

    for row in indices:
        rankings.append(
            [doc_ids[int(i)] for i in row]
        )

    return rankings


print("\nRetrieving MiniLM...")
val_dense = dense_retrieve(
    val,
    save_query_embeddings=True,
    prefix="val_minilm"
)

eval_dense = dense_retrieve(
    eval_claims,
    save_query_embeddings=True,
    prefix="eval_minilm"
)

print(
    f"MiniLM validation nDCG@10: "
    f"{evaluate(val_dense, val):.4f}"
)


# ============================================================
# RRF HYBRID
# ============================================================

def rrf_fuse(rank_a, rank_b, k):
    scores = {}

    for rank, doc_id in enumerate(rank_a, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (
            1.0 / (k + rank)
        )

    for rank, doc_id in enumerate(rank_b, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (
            1.0 / (k + rank)
        )

    ranked = sorted(
        scores,
        key=lambda d: scores[d],
        reverse=True
    )

    return ranked[:100]


print("\n" + "=" * 70)
print("TUNING RRF")
print("=" * 70)

best_rrf_k = None
best_rrf_score = -1
best_val_hybrid = None
best_eval_hybrid = None

for k in RRF_K_VALUES:

    val_hybrid = [
        rrf_fuse(b, d, k)
        for b, d in zip(val_bm25, val_dense)
    ]

    score = evaluate(
        val_hybrid,
        val
    )

    print(
        f"RRF k={k:3d} -> "
        f"validation nDCG@10 = {score:.4f}"
    )

    if score > best_rrf_score:
        best_rrf_score = score
        best_rrf_k = k
        best_val_hybrid = val_hybrid


best_eval_hybrid = [
    rrf_fuse(b, d, best_rrf_k)
    for b, d in zip(eval_bm25, eval_dense)
]

print(
    f"\nBEST HYBRID: RRF k={best_rrf_k}"
)

print(
    f"Best hybrid validation nDCG@10: "
    f"{best_rrf_score:.4f}"
)


# ============================================================
# CHOOSE BEST FIRST-STAGE SYSTEM
# ============================================================

bm25_score = evaluate(val_bm25, val)
dense_score = evaluate(val_dense, val)

first_stage_candidates = {
    "bm25": (bm25_score, val_bm25, eval_bm25),
    "minilm": (dense_score, val_dense, eval_dense),
    "hybrid": (
        best_rrf_score,
        best_val_hybrid,
        best_eval_hybrid
    )
}

best_first_stage_name = max(
    first_stage_candidates,
    key=lambda x: first_stage_candidates[x][0]
)

best_first_stage_score, best_first_stage_val, best_first_stage_eval = (
    first_stage_candidates[best_first_stage_name]
)

print("\n" + "=" * 70)
print("FIRST-STAGE SELECTION")
print("=" * 70)

for name, (score, _, _) in first_stage_candidates.items():
    print(
        f"{name:10s}: {score:.4f}"
    )

print(
    f"\nSelected first-stage system: "
    f"{best_first_stage_name}"
)


# ============================================================
# CROSS-ENCODER
# ============================================================

print("\nLoading cross-encoder...")
reranker = CrossEncoder(
    RERANKER_MODEL,
    max_length=512
)


def rerank_one(claim_text, first_stage_docs, k):
    top_docs = first_stage_docs[:k]

    pairs = []

    for doc_id in top_docs:
        idx = doc_id_to_idx[doc_id]
        pairs.append(
            (claim_text, documents[idx])
        )

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    order = np.argsort(
        np.asarray(scores)
    )[::-1]

    reranked_top = [
        top_docs[int(i)]
        for i in order
    ]

    remaining = [
        d for d in first_stage_docs
        if d not in set(reranked_top)
    ]

    return reranked_top + remaining


# ============================================================
# RERANK DEPTH TUNING
# ============================================================

print("\n" + "=" * 70)
print("TUNING CROSS-ENCODER RERANK DEPTH")
print("=" * 70)

best_rerank_k = None
best_rerank_score = -1
best_val_rerank = None

for k in RERANK_K_VALUES:

    print(
        f"\nReranking top-{k}..."
    )

    t0 = time.perf_counter()

    rankings = []

    for claim, first_stage in zip(
        val,
        best_first_stage_val
    ):
        rankings.append(
            rerank_one(
                claim["claim"],
                first_stage,
                k
            )[:100]
        )

    elapsed = time.perf_counter() - t0

    score = evaluate(
        rankings,
        val
    )

    per_query_ms = (
        elapsed / len(val)
    ) * 1000.0

    print(
        f"top-{k}: nDCG@10={score:.4f} | "
        f"{per_query_ms:.2f} ms/query"
    )

    if score > best_rerank_score:
        best_rerank_score = score
        best_rerank_k = k
        best_val_rerank = rankings
        best_rerank_latency_ms = per_query_ms


# ============================================================
# FINAL EVAL RERANK
# ============================================================

print(
    f"\nBEST RERANK DEPTH: top-{best_rerank_k}"
)

print(
    f"Validation nDCG@10: "
    f"{best_rerank_score:.4f}"
)

print("\nReranking evaluation claims...")

t0 = time.perf_counter()

final_eval_rerank = []

for claim, first_stage in zip(
    eval_claims,
    best_first_stage_eval
):
    final_eval_rerank.append(
        rerank_one(
            claim["claim"],
            first_stage,
            best_rerank_k
        )[:100]
    )

eval_elapsed = time.perf_counter() - t0

eval_latency_ms = (
    eval_elapsed / len(eval_claims)
) * 1000.0


# ============================================================
# WRITE REQUIRED RUNS
# ============================================================

print("\nWriting TREC files...")

write_trec(
    f"{PRED_DIR}/t3_hybrid.trec",
    eval_claims,
    best_eval_hybrid,
    "t3_hybrid"
)

write_trec(
    f"{PRED_DIR}/t3_rerank.trec",
    eval_claims,
    final_eval_rerank,
    "t3_rerank"
)


# ============================================================
# SAVE TASK 3 SUMMARY
# ============================================================

summary = {
    "bm25_val_ndcg10": bm25_score,
    "minilm_val_ndcg10": dense_score,
    "rrf_k_candidates": RRF_K_VALUES,
    "best_rrf_k": best_rrf_k,
    "hybrid_val_ndcg10": best_rrf_score,
    "best_first_stage": best_first_stage_name,
    "reranker_model": RERANKER_MODEL,
    "rerank_k_candidates": RERANK_K_VALUES,
    "best_rerank_k": best_rerank_k,
    "rerank_val_ndcg10": best_rerank_score,
    "rerank_val_latency_ms": best_rerank_latency_ms,
    "rerank_eval_latency_ms": eval_latency_ms,
    "dense_encoding_time_sec": encoding_time
}

with open(
    f"{ART_DIR}/task3_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# CLEANUP
# ============================================================

del reranker
del dense_model
del index

gc.collect()

print("\n" + "=" * 70)
print("TASK 3 COMPLETE")
print("=" * 70)

print(
    f"BM25       : {bm25_score:.4f}"
)

print(
    f"MiniLM     : {dense_score:.4f}"
)

print(
    f"Hybrid      : {best_rrf_score:.4f}"
)

print(
    f"Reranked    : {best_rerank_score:.4f}"
)

print(
    f"Best first-stage: {best_first_stage_name}"
)

print(
    f"Best rerank k  : {best_rerank_k}"
)

print(
    f"Rerank latency : "
    f"{best_rerank_latency_ms:.2f} ms/query"
)

print("\nGenerated:")
print("  predictions/t3_hybrid.trec")
print("  predictions/t3_rerank.trec")
print("  artifacts/task3_summary.json")
print("  artifacts/minilm_doc_embeddings.npy")
print("  artifacts/minilm_doc_ids.npy")

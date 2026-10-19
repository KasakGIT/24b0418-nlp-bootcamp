import json
import math
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PRED_DIR = BASE_DIR / "predictions"

PRED_DIR.mkdir(exist_ok=True)


# ============================================================
# Loading data
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
val_claims = load_jsonl(DATA_DIR / "val_claims.jsonl")
eval_claims = load_jsonl(DATA_DIR / "eval_claims.jsonl")

print(f"Corpus documents: {len(corpus)}")
print(f"Validation claims: {len(val_claims)}")
print(f"Evaluation claims: {len(eval_claims)}")


# ============================================================
# Tokenization
# ============================================================

def tokenize(text):
    return text.lower().split()


# Combine title + abstract into one searchable document.
doc_ids = []
documents = []

for doc in corpus:
    doc_ids.append(str(doc["doc_id"]))

    abstract_text = " ".join(doc["abstract"])
    full_text = doc["title"] + " " + abstract_text

    documents.append(tokenize(full_text))


# ============================================================
# Build BM25
# ============================================================

print("\nBuilding BM25 index...")

start = time.time()

bm25 = BM25Okapi(documents)

index_time = time.time() - start

print(f"BM25 index built in {index_time:.2f} seconds")


# ============================================================
# Retrieve top 100 documents
# ============================================================

def retrieve(claim, k1, b, top_k=100):
    """
    rank_bm25 stores k1 and b internally.
    We create a BM25 object for each parameter combination
    during tuning.
    """
    query_tokens = tokenize(claim)

    scores = bm25.get_scores(query_tokens)

    # Sort by descending score.
    ranked_indices = np.argsort(-scores, kind="stable")[:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        results.append(
            (
                doc_ids[idx],
                float(scores[idx]),
                rank
            )
        )

    return results


# ============================================================
# nDCG@10
# ============================================================

def ndcg_at_10(ranked_doc_ids, gold_doc_ids):
    """
    Binary relevance:
        1 = gold evidence document
        0 = not evidence

    Only top 10 documents are considered.
    """

    dcg = 0.0

    for rank, doc_id in enumerate(ranked_doc_ids[:10], start=1):
        if doc_id in gold_doc_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_relevant = min(len(gold_doc_ids), 10)

    if ideal_relevant == 0:
        return None

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )

    return dcg / idcg


# ============================================================
# Evaluate a BM25 configuration
# ============================================================

def evaluate_bm25(val_claims, k1, b):

    # Create BM25 with these parameters.
    model = BM25Okapi(
        documents,
        k1=k1,
        b=b
    )

    scores = []

    for item in val_claims:

        # Claims without evidence cannot be used to measure
        # retrieval quality.
        if not item["evidence"]:
            continue

        query_tokens = tokenize(item["claim"])

        doc_scores = model.get_scores(query_tokens)

        ranked_indices = np.argsort(
            -doc_scores,
            kind="stable"
        )[:100]

        ranked_doc_ids = [
            doc_ids[i]
            for i in ranked_indices
        ]

        gold_doc_ids = {
            str(e["doc_id"])
            for e in item["evidence"]
        }

        score = ndcg_at_10(
            ranked_doc_ids,
            gold_doc_ids
        )

        if score is not None:
            scores.append(score)

    return float(np.mean(scores))


# ============================================================
# Tune k1 and b
# ============================================================

print("\nTuning BM25...")

k1_values = [0.8, 1.2, 1.6, 2.0]
b_values = [0.3, 0.5, 0.75, 1.0]

best_score = -1
best_k1 = None
best_b = None

for k1 in k1_values:

    for b in b_values:

        score = evaluate_bm25(
            val_claims,
            k1,
            b
        )

        print(
            f"k1={k1:<3} "
            f"b={b:<4} "
            f"nDCG@10={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_k1 = k1
            best_b = b


print("\n==============================")
print("BEST BM25")
print("==============================")
print(f"k1       : {best_k1}")
print(f"b        : {best_b}")
print(f"nDCG@10  : {best_score:.4f}")
print("==============================")


# ============================================================
# Generate TREC file
# ============================================================

def write_trec(claims, output_path, run_name):

    model = BM25Okapi(
        documents,
        k1=best_k1,
        b=best_b
    )

    with open(output_path, "w", encoding="utf-8") as f:

        for item in tqdm(claims, desc=f"Writing {run_name}"):

            claim_id = str(item["id"])

            query_tokens = tokenize(item["claim"])

            scores = model.get_scores(query_tokens)

            ranked_indices = np.argsort(
                -scores,
                kind="stable"
            )[:100]

            for rank, idx in enumerate(
                ranked_indices,
                start=1
            ):

                doc_id = doc_ids[idx]
                score = float(scores[idx])

                f.write(
                    f"{claim_id} Q0 {doc_id} "
                    f"{rank} {score:.8f} {run_name}\n"
                )


# Validation run
write_trec(
    val_claims,
    PRED_DIR / "val_bm25.trec",
    "val_bm25"
)

# Evaluation run
write_trec(
    eval_claims,
    PRED_DIR / "t1_bm25.trec",
    "t1_bm25"
)


print("\nDone!")
print("Created:")
print("  predictions/val_bm25.trec")
print("  predictions/t1_bm25.trec")
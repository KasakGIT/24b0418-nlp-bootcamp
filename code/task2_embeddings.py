import json
import time
from pathlib import Path

import faiss
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRED = ROOT / "predictions"
PRED.mkdir(exist_ok=True)

CORPUS_FILE = DATA / "corpus.jsonl"
VAL_FILE = DATA / "val_claims.jsonl"
EVAL_FILE = DATA / "eval_claims.jsonl"


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def doc_text(doc):
    abstract = doc["abstract"]
    if isinstance(abstract, list):
        abstract = " ".join(abstract)
    return doc["title"] + " " + abstract


def dcg(rels):
    return sum(
        rel / np.log2(i + 2)
        for i, rel in enumerate(rels)
    )


def ndcg_at_10(results, claims_by_id, doc_ids):
    scores = []

    for claim in claims_by_id:
        gold = {
            str(e["doc_id"])
            for e in claim.get("evidence", [])
        }

        if not gold:
            continue

        ranked = results[str(claim["id"])][:10]

        rels = [
            1 if str(doc_ids[idx]) in gold else 0
            for idx in ranked
        ]

        ideal_len = min(len(gold), 10)
        ideal = [1] * ideal_len + [0] * (10 - ideal_len)

        actual_dcg = dcg(rels)
        ideal_dcg = dcg(ideal)

        scores.append(
            actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
        )

    return float(np.mean(scores))


def write_trec(results, claims, doc_ids, run_name, filename):
    path = PRED / filename

    with open(path, "w") as f:
        for claim in claims:
            cid = str(claim["id"])
            ranked = results[cid][:100]

            # Small epsilon guarantees strictly/non-increasing scores
            for rank, idx in enumerate(ranked, start=1):
                score = 1.0 / rank
                f.write(
                    f"{cid} Q0 {doc_ids[idx]} {rank} "
                    f"{score:.8f} {run_name}\n"
                )

    print(f"Wrote {path}")


def run_model(model_name, model_label, corpus, val_claims, eval_claims):
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")

    process = psutil.Process()

    print("Loading model...")
    model = SentenceTransformer(model_name, device="cpu")

    texts = [doc_text(d) for d in corpus]
    doc_ids = [d["doc_id"] for d in corpus]

    # ---------------------------------------------------------
    # Indexing
    # ---------------------------------------------------------
    print("Encoding corpus...")
    start = time.perf_counter()

    doc_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    index_time = time.perf_counter() - start

    print(f"Embedding dimension: {doc_embeddings.shape[1]}")
    print(f"Index/encoding time: {index_time:.2f} sec")

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    memory_mb = process.memory_info().rss / (1024 * 1024)

    # ---------------------------------------------------------
    # Validation retrieval
    # ---------------------------------------------------------
    val_texts = [c["claim"] for c in val_claims]

    start = time.perf_counter()

    val_embeddings = model.encode(
        val_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    distances, indices = index.search(val_embeddings, 100)

    total_query_time = time.perf_counter() - start
    latency_ms = (total_query_time / len(val_claims)) * 1000

    val_results = {
        str(claim["id"]): indices[i].tolist()
        for i, claim in enumerate(val_claims)
    }

    score = ndcg_at_10(
        val_results,
        val_claims,
        doc_ids,
    )

    print(f"Validation nDCG@10: {score:.4f}")
    print(f"Query latency: {latency_ms:.2f} ms/query")
    print(f"Memory RSS: {memory_mb:.2f} MB")

    # ---------------------------------------------------------
    # Evaluation retrieval
    # ---------------------------------------------------------
    eval_texts = [c["claim"] for c in eval_claims]

    eval_embeddings = model.encode(
        eval_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    _, eval_indices = index.search(eval_embeddings, 100)

    eval_results = {
        str(claim["id"]): eval_indices[i].tolist()
        for i, claim in enumerate(eval_claims)
    }

    return {
        "val_results": val_results,
        "eval_results": eval_results,
        "doc_ids": doc_ids,
        "val_ndcg10": score,
        "index_time_sec": index_time,
        "query_latency_ms": latency_ms,
        "memory_mb": memory_mb,
    }


def main():
    corpus = load_jsonl(CORPUS_FILE)
    val_claims = load_jsonl(VAL_FILE)
    eval_claims = load_jsonl(EVAL_FILE)

    print(f"Corpus: {len(corpus)}")
    print(f"Validation claims: {len(val_claims)}")
    print(f"Evaluation claims: {len(eval_claims)}")

    general = run_model(
        "sentence-transformers/all-MiniLM-L6-v2",
        "general",
        corpus,
        val_claims,
        eval_claims,
    )

    write_trec(
        general["val_results"],
        val_claims,
        general["doc_ids"],
        "val_general",
        "val_general.trec",
    )

    write_trec(
        general["eval_results"],
        eval_claims,
        general["doc_ids"],
        "t2_general",
        "t2_general.trec",
    )

    scientific = run_model(
        "allenai-specter",
        "scientific",
        corpus,
        val_claims,
        eval_claims,
    )

    write_trec(
        scientific["eval_results"],
        eval_claims,
        scientific["doc_ids"],
        "t2_scientific",
        "t2_scientific.trec",
    )

    print("\n" + "=" * 60)
    print("TASK 2 SUMMARY")
    print("=" * 60)

    print(
        f"General nDCG@10:     {general['val_ndcg10']:.4f}"
    )
    print(
        f"Scientific nDCG@10:  {scientific['val_ndcg10']:.4f}"
    )

    print(
        f"General index time:  {general['index_time_sec']:.2f}s"
    )
    print(
        f"Scientific index time:{scientific['index_time_sec']:.2f}s"
    )

    print(
        f"General latency:     {general['query_latency_ms']:.2f} ms"
    )
    print(
        f"Scientific latency:  {scientific['query_latency_ms']:.2f} ms"
    )

    print(
        f"General memory:      {general['memory_mb']:.2f} MB"
    )
    print(
        f"Scientific memory:   {scientific['memory_mb']:.2f} MB"
    )


if __name__ == "__main__":
    main()

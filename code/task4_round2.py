#!/usr/bin/env python3
"""
Task 4 - Competitive Round 2
SciFact bi-encoder fine-tuning with:
  1) hard negatives mined from BOTH BM25 and MiniLM
  2) MultipleNegativesRankingLoss with explicit hard negatives
  3) small validation-driven hyperparameter sweep

This is an additional experiment on top of the original Task 4 script.
It does NOT use validation/eval labels for training or negative mining.

Run from the NLP project root:
    python code/task4_round2.py
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers import losses
from sentence_transformers.training_args import BatchSamplers


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRED = ROOT / "predictions"
ART = ROOT / "artifacts"
OUT = ART / "task4_round2"

PRED.mkdir(exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_PATH = ART / "minilm_doc_embeddings.npy"
DOC_ID_PATH = ART / "minilm_doc_ids.npy"

# Small, targeted sweep: do not waste CPU on a huge grid.
CONFIGS = [
    # (learning rate, epochs, batch size, number of explicit hard negatives)
    {"lr": 5e-6, "epochs": 1, "batch": 16, "num_neg": 4},
    {"lr": 1e-5, "epochs": 1, "batch": 16, "num_neg": 4},
    {"lr": 5e-6, "epochs": 2, "batch": 16, "num_neg": 4},
    {"lr": 1e-5, "epochs": 1, "batch": 8,  "num_neg": 4},
]

MAX_CANDIDATES = 100
MINE_TOP_BM25 = 50
MINE_TOP_DENSE = 50


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def claim_text(row):
    return row["claim"].strip()


def doc_text(doc):
    abstract = doc.get("abstract", [])
    if isinstance(abstract, list):
        abstract = " ".join(abstract)
    return f'{doc.get("title", "")} {abstract}'.strip()


def tokenize(text):
    return text.lower().split()


def gold_doc_ids(row):
    out = set()
    for ev in row.get("evidence", []):
        if isinstance(ev, dict) and "doc_id" in ev:
            out.add(str(ev["doc_id"]))
    return out


def ndcg10_binary(rows, rankings):
    vals = []
    for row, ranked in zip(rows, rankings):
        gold = gold_doc_ids(row)
        if not gold:
            continue

        dcg = 0.0
        for rank, doc_id in enumerate(ranked[:10], 1):
            if str(doc_id) in gold:
                dcg += 1.0 / np.log2(rank + 1)

        ideal_n = min(len(gold), 10)
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_n + 1))
        vals.append(dcg / idcg if idcg else 0.0)

    return float(np.mean(vals)) if vals else 0.0


def rank_all(model, texts, corpus_texts, batch_size=32):
    q = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    d = model.encode(
        corpus_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    scores = q @ d.T
    order = np.argsort(-scores, axis=1)
    return order


def write_trec(path, rows, rankings, doc_ids, run_name):
    with open(path, "w", encoding="utf-8") as f:
        for row, order in zip(rows, rankings):
            cid = str(row["id"])
            for rank, idx in enumerate(order[:100], 1):
                score = 1.0 / rank
                f.write(
                    f"{cid} Q0 {doc_ids[int(idx)]} {rank} "
                    f"{score:.8f} {run_name}\n"
                )


def build_training_pairs(train_rows, corpus_by_id, bm25, doc_ids, dense_matrix):
    """
    For every training claim with evidence:
      - take top BM25 candidates
      - take top MiniLM candidates
      - exclude all gold docs
      - build a diverse pool from both systems
    """
    doc_id_to_idx = {str(d): i for i, d in enumerate(doc_ids)}
    all_doc_ids = [str(x) for x in doc_ids]

    pairs = []
    stats = {
        "claims_with_evidence": 0,
        "claims_with_4_negatives": 0,
        "bm25_only": 0,
        "dense_only": 0,
        "overlap": 0,
    }

    for row in train_rows:
        gold = gold_doc_ids(row)
        if not gold:
            continue

        positives = [d for d in gold if d in corpus_by_id]
        if not positives:
            continue

        stats["claims_with_evidence"] += 1
        pos_id = positives[0]

        q = claim_text(row)

        # BM25 candidates
        bm25_scores = bm25.get_scores(tokenize(q))
        bm25_order = np.argsort(-bm25_scores)[:MAX_CANDIDATES]
        bm25_candidates = [
            str(all_doc_ids[int(i)])
            for i in bm25_order
            if str(all_doc_ids[int(i)]) not in gold
        ][:MINE_TOP_BM25]

        # MiniLM candidates from the saved Task-3 corpus embeddings.
        # This avoids recomputing 5,183 document embeddings.
        q_emb = dense_matrix[int(row["_train_dense_idx"])]
        dense_scores = dense_matrix @ q_emb
        dense_order = np.argsort(-dense_scores)[:MAX_CANDIDATES]
        dense_candidates = [
            str(all_doc_ids[int(i)])
            for i in dense_order
            if str(all_doc_ids[int(i)]) not in gold
        ][:MINE_TOP_DENSE]

        bm = set(bm25_candidates)
        de = set(dense_candidates)
        stats["overlap"] += len(bm & de)

        # Deliberately take candidates from both retrieval behaviours.
        negatives = []
        used = set()

        for i in range(max(len(bm25_candidates), len(dense_candidates))):
            if i < len(dense_candidates):
                d = dense_candidates[i]
                if d not in used:
                    negatives.append(d)
                    used.add(d)

            if i < len(bm25_candidates):
                d = bm25_candidates[i]
                if d not in used:
                    negatives.append(d)
                    used.add(d)

            if len(negatives) >= 4:
                break

        if len(negatives) < 4:
            # Fill from dense then BM25, then deterministic corpus fallback.
            for source in (dense_candidates, bm25_candidates):
                for d in source:
                    if d not in used:
                        negatives.append(d)
                        used.add(d)
                    if len(negatives) >= 4:
                        break
                if len(negatives) >= 4:
                    break

        if len(negatives) < 4:
            for d in all_doc_ids:
                if d not in gold and d not in used:
                    negatives.append(d)
                    used.add(d)
                if len(negatives) >= 4:
                    break

        if len(negatives) >= 4:
            stats["claims_with_4_negatives"] += 1

        # Keep up to four negatives. The sweep may use fewer if desired.
        pairs.append(
            {
                "anchor": q,
                "positive": doc_text(corpus_by_id[pos_id]),
                "negatives": negatives[:4],
                "positive_doc_id": pos_id,
            }
        )

    stats["bm25_dense_overlap_avg"] = (
        stats["overlap"] / stats["claims_with_evidence"]
        if stats["claims_with_evidence"]
        else 0
    )
    return pairs, stats


def make_dataset(pairs, num_neg):
    data = {
        "anchor": [],
        "positive": [],
    }
    for j in range(num_neg):
        data[f"negative_{j+1}"] = []

    for p in pairs:
        if len(p["negatives"]) < num_neg:
            continue
        data["anchor"].append(p["anchor"])
        data["positive"].append(p["positive"])
        for j in range(num_neg):
            data[f"negative_{j+1}"].append(p["negatives"][j])

    return Dataset.from_dict(data)


def train_one(cfg, train_pairs, val_rows, corpus_texts, doc_ids, run_dir):
    print("\n" + "=" * 80)
    print("CONFIG:", cfg)
    print("=" * 80)

    model = SentenceTransformer(MODEL_NAME)

    train_ds = make_dataset(train_pairs, cfg["num_neg"])

    # MultipleNegativesRankingLoss supports (anchor, positive, negative_1, ...)
    # and additionally benefits from in-batch negatives.
    loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=20.0,
        )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=cfg["batch"],
        learning_rate=cfg["lr"],
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=False,
        bf16=False,
        logging_steps=25,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        seed=SEED,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        loss=loss,
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    # Evaluate ONLY after training, using the provided validation split.
    val_claims = [claim_text(x) for x in val_rows]
    q_emb = model.encode(
        val_claims,
        batch_size=cfg["batch"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    d_emb = model.encode(
        corpus_texts,
        batch_size=cfg["batch"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    scores = q_emb @ d_emb.T
    order = np.argsort(-scores, axis=1)
    ranked_doc_ids = [[str(doc_ids[int(i)]) for i in row[:100]] for row in order]
    score = ndcg10_binary(val_rows, ranked_doc_ids)

    print(f"Validation nDCG@10: {score:.4f}")
    print(f"Training time: {train_seconds:.1f}s")

    return model, score, train_seconds, ranked_doc_ids


def main():
    print("Task 4 Round 2 starting...")
    print("Project root:", ROOT)

    train_rows = load_jsonl(DATA / "train_claims.jsonl")
    val_rows = load_jsonl(DATA / "val_claims.jsonl")
    eval_rows = load_jsonl(DATA / "eval_claims.jsonl")
    corpus = load_jsonl(DATA / "corpus.jsonl")

    corpus_by_id = {str(d["doc_id"]): d for d in corpus}
    doc_ids = np.load(DOC_ID_PATH, allow_pickle=True)
    doc_ids = [str(x) for x in doc_ids]

    dense_matrix = np.load(EMB_PATH)
    dense_matrix = np.asarray(dense_matrix, dtype=np.float32)

    if len(doc_ids) != len(corpus):
        raise RuntimeError("Saved MiniLM doc IDs do not match corpus size.")

    # Train-claim dense embeddings are NOT saved by Task 3, so compute only
    # the 647 query embeddings once for hard-negative mining.
    base_model = SentenceTransformer(MODEL_NAME)
    train_claim_texts = [claim_text(x) for x in train_rows]
    train_q = base_model.encode(
        train_claim_texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # Store an internal index into train_q, never exposed as dataset content.
    for i, row in enumerate(train_rows):
        row["_train_dense_idx"] = i

    corpus_texts = [doc_text(d) for d in corpus]

    print("Building BM25...")
    bm25 = BM25Okapi([tokenize(x) for x in corpus_texts])

    print("Mining mixed BM25 + MiniLM hard negatives...")
    pairs = []
    overlap_total = 0
    claims_with_evidence = 0

    for i, row in enumerate(train_rows):
        gold = gold_doc_ids(row)
        if not gold:
            continue
        positives = [d for d in gold if d in corpus_by_id]
        if not positives:
            continue

        claims_with_evidence += 1
        pos_id = positives[0]

        # BM25
        b_scores = bm25.get_scores(tokenize(claim_text(row)))
        b_order = np.argsort(-b_scores)[:MAX_CANDIDATES]
        b_cands = [
            str(doc_ids[int(j)]) for j in b_order
            if str(doc_ids[int(j)]) not in gold
        ][:MINE_TOP_BM25]

        # Dense
        d_scores = dense_matrix @ train_q[i]
        d_order = np.argsort(-d_scores)[:MAX_CANDIDATES]
        d_cands = [
            str(doc_ids[int(j)]) for j in d_order
            if str(doc_ids[int(j)]) not in gold
        ][:MINE_TOP_DENSE]

        overlap_total += len(set(b_cands) & set(d_cands))

        negs, used = [], set()
        for j in range(max(len(b_cands), len(d_cands))):
            if j < len(d_cands) and d_cands[j] not in used:
                negs.append(d_cands[j]); used.add(d_cands[j])
            if j < len(b_cands) and b_cands[j] not in used:
                negs.append(b_cands[j]); used.add(b_cands[j])
            if len(negs) >= 4:
                break

        if len(negs) < 4:
            for source in (d_cands, b_cands):
                for d in source:
                    if d not in used:
                        negs.append(d); used.add(d)
                    if len(negs) >= 4:
                        break
                if len(negs) >= 4:
                    break

        if len(negs) < 4:
            for d in doc_ids:
                if d not in gold and d not in used:
                    negs.append(d); used.add(d)
                if len(negs) >= 4:
                    break

        if len(negs) >= 4:
            pairs.append({
                "anchor": claim_text(row),
                "positive": doc_text(corpus_by_id[pos_id]),
                "negatives": negs[:4],
                "positive_doc_id": pos_id,
            })

    print(f"Training claims with evidence: {claims_with_evidence}")
    print(f"Usable mixed-hard examples: {len(pairs)}")
    print(
        "Average BM25/MiniLM candidate overlap:",
        round(overlap_total / max(1, claims_with_evidence), 2),
    )

    # Save mining metadata.
    with open(OUT / "mining_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "bm25_top": MINE_TOP_BM25,
                "dense_top": MINE_TOP_DENSE,
                "num_negatives": 4,
                "train_claims_with_evidence": claims_with_evidence,
                "usable_examples": len(pairs),
                "avg_bm25_dense_overlap": overlap_total / max(1, claims_with_evidence),
            },
            f,
            indent=2,
        )

    results = []
    best = None

    for idx, cfg in enumerate(CONFIGS, 1):
        run_dir = OUT / f"config_{idx}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model, val_score, train_seconds, val_rankings = train_one(
            cfg,
            pairs,
            val_rows,
            corpus_texts,
            doc_ids,
            run_dir,
        )

        rec = {
            "config_id": idx,
            **cfg,
            "val_ndcg10": val_score,
            "train_seconds": train_seconds,
        }
        results.append(rec)

        if best is None or val_score > best["val_ndcg10"]:
            # Save best model immediately.
            best_dir = OUT / "best_model"
            if best_dir.exists():
                import shutil
                shutil.rmtree(best_dir)
            model.save(str(best_dir))

            best = {
                **rec,
                "model_dir": str(best_dir),
                "val_rankings": val_rankings,
            }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("ROUND 2 RESULTS")
    print("=" * 80)
    for r in results:
        print(
            f"config {r['config_id']}: "
            f"lr={r['lr']} epochs={r['epochs']} batch={r['batch']} "
            f"neg={r['num_neg']} -> {r['val_ndcg10']:.4f}"
        )

    print(f"\nBEST ROUND 2: {best['val_ndcg10']:.4f}")

    # Load best model and generate final required Task-4 hard run.
    best_model = SentenceTransformer(best["model_dir"])

    def retrieve(rows):
        q = best_model.encode(
            [claim_text(x) for x in rows],
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        d = best_model.encode(
            corpus_texts,
            batch_size=16,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        s = q @ d.T
        o = np.argsort(-s, axis=1)
        return o

    eval_rankings = retrieve(eval_rows)

    # Only replace t4_hard if Round 2 beats the original Task-4 hard score.
    original_hard = 0.8327
    replaced = best["val_ndcg10"] > original_hard

    if replaced:
        write_trec(
            PRED / "t4_hard.trec",
            eval_rows,
            eval_rankings,
            doc_ids,
            "t4_hard",
        )
        print("Round 2 BEATS original 0.8327 -> t4_hard.trec UPDATED.")
    else:
        print(
            "Round 2 did not beat original 0.8327. "
            "Keeping the original t4_hard.trec."
        )

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "configs": results,
                "best": {
                    k: v for k, v in best.items()
                    if k != "val_rankings"
                },
                "original_task4_hard_val_ndcg10": original_hard,
                "replaced_required_t4_hard": replaced,
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print("  artifacts/task4_round2/mining_summary.json")
    print("  artifacts/task4_round2/summary.json")
    print("  artifacts/task4_round2/best_model/")
    if replaced:
        print("  predictions/t4_hard.trec")


if __name__ == "__main__":
    main()

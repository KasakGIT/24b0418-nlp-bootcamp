#!/usr/bin/env python3
"""Measure validation gold-evidence coverage in retrieval runs (no model downloads).

Run from project root:
    python code/task5_retrieval_coverage.py

Reports both:
- claim coverage: fraction of evidence-bearing claims with >=1 gold evidence doc retrieved
- evidence-document recall: fraction of distinct gold evidence docs retrieved

This is a validation diagnostic only; it does not change predictions.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PRED = ROOT / "predictions"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_trec(path):
    runs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 6:
                raise ValueError(f"Malformed TREC line in {path}: {line.rstrip()}")
            qid, docid, rank = int(p[0]), int(p[2]), int(p[3])
            runs[qid].append((rank, docid))
    return {qid: [docid for rank, docid in sorted(rows)][:100]
            for qid, rows in runs.items()}


def gold_docs(claim):
    docs = set()
    for ev in claim.get("evidence", []) or []:
        if isinstance(ev, dict) and "doc_id" in ev and str(ev.get("label", "")).upper() in {"SUPPORT", "CONTRADICT"}:
            docs.add(int(ev["doc_id"]))
    return docs


def make_rrf(bm25, dense, claim_ids, k=15, depth=100):
    fused = {}
    for cid in claim_ids:
        br = bm25.get(cid, [])
        dr = dense.get(cid, [])
        bpos = {did: i + 1 for i, did in enumerate(br)}
        dpos = {did: i + 1 for i, did in enumerate(dr)}
        candidates = set(br) | set(dr)
        fused[cid] = sorted(
            candidates,
            key=lambda did: (
                -(1 / (k + bpos.get(did, 10**6)) + 1 / (k + dpos.get(did, 10**6))),
                did,
            ),
        )[:depth]
    return fused


def report(name, rankings, claims, ks=(10, 100)):
    eligible = {int(c["id"]): gold_docs(c) for c in claims}
    eligible = {cid: docs for cid, docs in eligible.items() if docs}
    total_gold_docs = sum(len(docs) for docs in eligible.values())
    print(f"\n{name}")
    print(f"Evidence-bearing claims: {len(eligible)}/{len(claims)}")
    for k in ks:
        hit_claims = 0
        found_gold_docs = 0
        for cid, docs in eligible.items():
            retrieved = set(rankings.get(cid, [])[:k])
            if docs & retrieved:
                hit_claims += 1
            found_gold_docs += len(docs & retrieved)
        claim_cov = hit_claims / len(eligible) if eligible else 0.0
        doc_recall = found_gold_docs / total_gold_docs if total_gold_docs else 0.0
        print(
            f"  Top-{k:>3}: claim coverage={hit_claims}/{len(eligible)} "
            f"({claim_cov:.1%}); evidence-doc recall={found_gold_docs}/{total_gold_docs} "
            f"({doc_recall:.1%})"
        )


def main():
    val_path = DATA / "val_claims.jsonl"
    bm25_path = PRED / "val_bm25.trec"
    dense_path = PRED / "val_general.trec"
    for path in (val_path, bm25_path, dense_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    claims = read_jsonl(val_path)
    claim_ids = [int(c["id"]) for c in claims]
    bm25 = read_trec(bm25_path)
    dense = read_trec(dense_path)
    rrf = make_rrf(bm25, dense, claim_ids, k=15, depth=100)

    print("Validation retrieval coverage diagnostic (no NLI model, no training)")
    print("Gold evidence documents are taken only from validation labels.")
    report("BM25", bm25, claims)
    report("General MiniLM dense retrieval", dense, claims)
    report("RRF union (k=15; same validation fallback used by Task 5)", rrf, claims)
    print("\nImportant: Task 5 currently scores only the top 10 documents per claim.")
    print("Compare RRF Top-10 coverage with Top-100 coverage to see what the verifier actually sees.")


if __name__ == "__main__":
    main()

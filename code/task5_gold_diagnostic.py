import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from task5_verify_finalize import gold_triples, read_jsonl

scores = json.loads(
    (ROOT / "artifacts/task5_val_pair_scores.json").read_text()
)

claims = read_jsonl(str(ROOT / "data/val_claims.jsonl"))

# gold_triples() takes ONE claim at a time.
gold = set()
for claim in claims:
    gold.update(gold_triples(claim))

print("Validation claims:", len(claims))
print("Gold evidence triples:", len(gold))

# Index cached predictions by (claim_id, doc_id).
by_pair = {}
for cid, rows in scores.items():
    for row in rows:
        by_pair[(int(cid), int(row["doc_id"]))] = row

counts = Counter()
neutral_errors = []
opposite_errors = []

for cid, did, gold_label in sorted(gold):
    row = by_pair.get((cid, did))

    if row is None:
        counts["not_in_top10"] += 1
        continue

    probs = {
        "SUPPORT": row["support_prob"],
        "CONTRADICT": row["contradict_prob"],
        "NEUTRAL": row["neutral_prob"],
    }

    predicted = max(probs, key=probs.get)
    counts["retrieved"] += 1

    if predicted == gold_label:
        counts["correct_label"] += 1
    elif predicted == "NEUTRAL":
        counts["neutral_instead"] += 1
        neutral_errors.append((cid, did, gold_label, probs))
    else:
        counts["opposite_label"] += 1
        opposite_errors.append((cid, did, gold_label, predicted, probs))

print("\nGold evidence results:")
for key, value in counts.items():
    print(f"{key}: {value}")

print("\nNEUTRAL instead of gold label (up to 10):")
for cid, did, gold_label, probs in neutral_errors[:10]:
    print(
        f"claim={cid} doc={did} gold={gold_label} "
        f"support={probs['SUPPORT']:.3f} "
        f"contradict={probs['CONTRADICT']:.3f} "
        f"neutral={probs['NEUTRAL']:.3f}"
    )

print("\nOpposite label (up to 10):")
for cid, did, gold_label, predicted, probs in opposite_errors[:10]:
    print(
        f"claim={cid} doc={did} gold={gold_label} predicted={predicted} "
        f"support={probs['SUPPORT']:.3f} "
        f"contradict={probs['CONTRADICT']:.3f} "
        f"neutral={probs['NEUTRAL']:.3f}"
    )
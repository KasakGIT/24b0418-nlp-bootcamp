import json
import os
import random
import shutil
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import TripletLoss
from torch.utils.data import DataLoader


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TRAIN_PATH = "data/train_claims.jsonl"
VAL_PATH = "data/val_claims.jsonl"
EVAL_PATH = "data/eval_claims.jsonl"
CORPUS_PATH = "data/corpus.jsonl"

PRED_DIR = "predictions"
ART_DIR = "artifacts"

TUNING_DIR = os.path.join(ART_DIR, "task4_tuning")

# Required final model directories
RANDOM_MODEL_DIR = os.path.join(
    ART_DIR, "task4_random_model"
)

HARD_MODEL_DIR = os.path.join(
    ART_DIR, "task4_hard_model"
)

# Hyperparameter grid
LEARNING_RATES = [1e-5, 2e-5]
EPOCHS_LIST = [1, 2]

BATCH_SIZE = 16
WARMUP_RATIO = 0.1
TRIPLET_MARGIN = 0.5

# Number of explicit negatives per positive
NUM_NEGATIVES = 2

# Hard-negative mining depth
HARD_TOP_K = 100


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_claim_id(claim):
    return str(claim["id"])


def get_gold_ids(claim):
    return {
        str(ev["doc_id"])
        for ev in claim.get("evidence", [])
    }


def build_doc_text(doc):
    title = doc.get("title", "")
    abstract = doc.get("abstract", [])

    if isinstance(abstract, list):
        abstract = " ".join(abstract)

    return f"{title}. {abstract}".strip()


def ndcg_at_10(rankings, claims):
    """
    Binary relevance nDCG@10.
    Only validation claims with at least one
    gold evidence abstract contribute.
    """

    claims_by_id = {
        get_claim_id(c): c
        for c in claims
    }

    values = []

    for claim_id, ranked_docs in rankings.items():

        claim = claims_by_id[claim_id]
        gold = get_gold_ids(claim)

        if not gold:
            continue

        dcg = 0.0

        for rank, (doc_id, _) in enumerate(
            ranked_docs[:10],
            start=1
        ):
            if str(doc_id) in gold:
                dcg += 1.0 / np.log2(rank + 1)

        ideal_hits = min(len(gold), 10)

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )

        if idcg > 0:
            values.append(dcg / idcg)

    return float(np.mean(values)) if values else 0.0


def save_trec(path, rankings, run_name):
    """
    TREC format:
    claim_id Q0 doc_id rank score run_name

    Scores are 1/rank so they are guaranteed
    to be non-increasing.
    """

    with open(path, "w", encoding="utf-8") as f:

        for claim_id in sorted(
            rankings.keys(),
            key=lambda x: int(x)
        ):

            docs = rankings[claim_id][:100]

            for rank, (doc_id, _) in enumerate(
                docs,
                start=1
            ):

                score = 1.0 / rank

                f.write(
                    f"{claim_id} Q0 {doc_id} "
                    f"{rank} {score:.8f} {run_name}\n"
                )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("TASK 4 — TUNED BI-ENCODER FINE-TUNING")
print("=" * 75)

print("\nLoading data...")

train_claims = load_jsonl(TRAIN_PATH)
val_claims = load_jsonl(VAL_PATH)
eval_claims = load_jsonl(EVAL_PATH)
corpus = load_jsonl(CORPUS_PATH)

print(f"Training claims  : {len(train_claims)}")
print(f"Validation claims: {len(val_claims)}")
print(f"Evaluation claims: {len(eval_claims)}")
print(f"Corpus documents : {len(corpus)}")


# ============================================================
# CORPUS
# ============================================================

doc_text = {}

for doc in corpus:
    doc_id = str(doc["doc_id"])
    doc_text[doc_id] = build_doc_text(doc)

all_doc_ids = list(doc_text.keys())

print(f"Usable documents : {len(all_doc_ids)}")


# ============================================================
# LOAD TASK 3 MINILM EMBEDDINGS
# ============================================================

print("\nLoading Task 3 MiniLM corpus embeddings...")

embedding_path = os.path.join(
    ART_DIR,
    "minilm_doc_embeddings.npy"
)

doc_ids_path = os.path.join(
    ART_DIR,
    "minilm_doc_ids.npy"
)

base_doc_embeddings = np.load(
    embedding_path
).astype(np.float32)

embedding_doc_ids = np.load(
    doc_ids_path,
    allow_pickle=True
)

embedding_doc_ids = [
    str(x)
    for x in embedding_doc_ids
]

print(
    "Embedding matrix:",
    base_doc_embeddings.shape
)


# ============================================================
# HARD NEGATIVE MINING
# ============================================================

print("\n" + "=" * 75)
print("MINING HARD NEGATIVES FROM TASK 2 MINILM")
print("=" * 75)

print(
    "Hard negatives are mined from the pretrained "
    "MiniLM retrieval system used in Task 2."
)

base_model = SentenceTransformer(MODEL_NAME)

train_query_texts = [
    str(c["claim"])
    for c in train_claims
]

train_query_embeddings = base_model.encode(
    train_query_texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True
)

train_query_embeddings = np.asarray(
    train_query_embeddings,
    dtype=np.float32
)

# Make sure corpus vectors are normalized
base_doc_embeddings = (
    base_doc_embeddings
    / np.maximum(
        np.linalg.norm(
            base_doc_embeddings,
            axis=1,
            keepdims=True
        ),
        1e-12
    )
)

index = faiss.IndexFlatIP(
    base_doc_embeddings.shape[1]
)

index.add(base_doc_embeddings)

_, retrieved_indices = index.search(
    train_query_embeddings,
    HARD_TOP_K
)


# ============================================================
# CREATE HARD NEGATIVE MAP
# ============================================================

hard_negative_map = {}

for i, claim in enumerate(train_claims):

    claim_id = get_claim_id(claim)
    gold = get_gold_ids(claim)

    hard_negatives = []

    for idx in retrieved_indices[i]:

        doc_id = embedding_doc_ids[idx]

        if doc_id in gold:
            continue

        if doc_id not in hard_negatives:
            hard_negatives.append(doc_id)

        if len(hard_negatives) >= NUM_NEGATIVES:
            break

    hard_negative_map[claim_id] = hard_negatives


num_with_hard = sum(
    bool(v)
    for v in hard_negative_map.values()
)

print(
    f"Claims with mined hard negatives: "
    f"{num_with_hard}/{len(train_claims)}"
)


# Free base model before training
del base_model
del train_query_embeddings
del index
del retrieved_indices


# ============================================================
# BUILD RANDOM-NEGATIVE TRIPLETS
# ============================================================

print("\n" + "=" * 75)
print("BUILDING RANDOM-NEGATIVE TRIPLETS")
print("=" * 75)

random_examples = []

rng = random.Random(SEED)

for claim in train_claims:

    claim_text = str(claim["claim"])
    gold = list(get_gold_ids(claim))

    if not gold:
        continue

    available_negatives = [
        d
        for d in all_doc_ids
        if d not in gold
    ]

    for positive_id in gold:

        positive_text = doc_text[positive_id]

        negatives = rng.sample(
            available_negatives,
            min(
                NUM_NEGATIVES,
                len(available_negatives)
            )
        )

        for negative_id in negatives:

            random_examples.append(
                InputExample(
                    texts=[
                        claim_text,
                        positive_text,
                        doc_text[negative_id]
                    ]
                )
            )

print(
    f"Random triplets: {len(random_examples)}"
)


# ============================================================
# BUILD HARD-NEGATIVE TRIPLETS
# ============================================================

print("\n" + "=" * 75)
print("BUILDING HARD-NEGATIVE TRIPLETS")
print("=" * 75)

hard_examples = []

for claim in train_claims:

    claim_id = get_claim_id(claim)
    claim_text = str(claim["claim"])

    gold = list(get_gold_ids(claim))

    if not gold:
        continue

    hard_negs = hard_negative_map.get(
        claim_id,
        []
    )

    for positive_id in gold:

        positive_text = doc_text[positive_id]

        for negative_id in hard_negs:

            hard_examples.append(
                InputExample(
                    texts=[
                        claim_text,
                        positive_text,
                        doc_text[negative_id]
                    ]
                )
            )

print(
    f"Hard-negative triplets: {len(hard_examples)}"
)


# ============================================================
# TRAIN ONE CONFIGURATION
# ============================================================

def train_configuration(
    examples,
    learning_rate,
    epochs,
    variant,
    run_number
):

    config_name = (
        f"{variant}_lr{learning_rate}_"
        f"epochs{epochs}"
    )

    output_dir = os.path.join(
        TUNING_DIR,
        config_name
    )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print(
        f"TRAINING CONFIG {run_number}"
    )
    print("=" * 75)

    print(f"Variant      : {variant}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs       : {epochs}")
    print(f"Batch size   : {BATCH_SIZE}")
    print(f"Triplet margin: {TRIPLET_MARGIN}")
    print(f"Triplets     : {len(examples)}")

    set_seed(SEED)

    model = SentenceTransformer(
        MODEL_NAME
    )

    dataloader = DataLoader(
        examples,
        shuffle=True,
        batch_size=BATCH_SIZE
    )

    train_loss = TripletLoss(
        model=model,
        triplet_margin=TRIPLET_MARGIN
    )

    warmup_steps = max(
        1,
        int(
            len(dataloader)
            * epochs
            * WARMUP_RATIO
        )
    )

    start = time.time()

    model.fit(
        train_objectives=[
            (dataloader, train_loss)
        ],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={
            "lr": learning_rate
        },
        show_progress_bar=True,
        use_amp=False
    )

    elapsed = time.time() - start

    model.save(output_dir)

    print(
        f"\nTraining time: "
        f"{elapsed / 60:.2f} minutes"
    )

    return model, output_dir, elapsed


# ============================================================
# RETRIEVE + VALIDATE
# ============================================================

def evaluate_validation(
    model,
    claims,
    variant,
    learning_rate,
    epochs
):

    print("\nEncoding corpus...")

    corpus_embeddings = model.encode(
        [
            doc_text[d]
            for d in all_doc_ids
        ],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    corpus_embeddings = np.asarray(
        corpus_embeddings,
        dtype=np.float32
    )

    index = faiss.IndexFlatIP(
        corpus_embeddings.shape[1]
    )

    index.add(corpus_embeddings)

    query_embeddings = model.encode(
        [
            str(c["claim"])
            for c in claims
        ],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype=np.float32
    )

    scores, indices = index.search(
        query_embeddings,
        100
    )

    rankings = {}

    for i, claim in enumerate(claims):

        claim_id = get_claim_id(claim)

        rankings[claim_id] = [
            (
                all_doc_ids[idx],
                float(scores[i][rank])
            )
            for rank, idx in enumerate(
                indices[i]
            )
        ]

    score = ndcg_at_10(
        rankings,
        claims
    )

    print(
        f"\nValidation nDCG@10: {score:.4f}"
    )

    # Save validation ranking for analysis
    val_path = os.path.join(
        TUNING_DIR,
        f"{variant}_lr{learning_rate}_"
        f"epochs{epochs}_val.trec"
    )

    save_trec(
        val_path,
        rankings,
        f"{variant}_tuned"
    )

    return score


# ============================================================
# TUNING LOOP
# ============================================================

results = []

configs = []

for lr in LEARNING_RATES:
    for epochs in EPOCHS_LIST:
        configs.append(
            (lr, epochs)
        )


total_configs = len(configs) * 2
run_number = 0


for variant, examples in [
    ("random", random_examples),
    ("hard", hard_examples)
]:

    for lr, epochs in configs:

        run_number += 1

        start = time.time()

        model, model_dir, train_time = (
            train_configuration(
                examples=examples,
                learning_rate=lr,
                epochs=epochs,
                variant=variant,
                run_number=run_number
            )
        )

        val_score = evaluate_validation(
            model=model,
            claims=val_claims,
            variant=variant,
            learning_rate=lr,
            epochs=epochs
        )

        results.append({
            "variant": variant,
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": BATCH_SIZE,
            "triplet_margin": TRIPLET_MARGIN,
            "train_triplets": len(examples),
            "validation_ndcg10": val_score,
            "training_time_seconds": train_time,
            "model_dir": model_dir
        })

        print(
            f"\nCONFIG RESULT "
            f"{run_number}/{total_configs}: "
            f"{variant} | "
            f"lr={lr} | "
            f"epochs={epochs} | "
            f"nDCG@10={val_score:.4f}"
        )

        del model


# ============================================================
# SELECT BEST CONFIGURATIONS
# ============================================================

random_results = [
    r
    for r in results
    if r["variant"] == "random"
]

hard_results = [
    r
    for r in results
    if r["variant"] == "hard"
]

best_random = max(
    random_results,
    key=lambda x: x["validation_ndcg10"]
)

best_hard = max(
    hard_results,
    key=lambda x: x["validation_ndcg10"]
)


print("\n" + "=" * 75)
print("TASK 4 TUNING RESULTS")
print("=" * 75)

print("\nALL CONFIGURATIONS:")

for r in results:

    print(
        f"{r['variant']:7s} | "
        f"lr={r['learning_rate']:<6g} | "
        f"epochs={r['epochs']} | "
        f"nDCG@10={r['validation_ndcg10']:.4f}"
    )


print("\nBEST RANDOM:")
print(json.dumps(best_random, indent=2))

print("\nBEST HARD:")
print(json.dumps(best_hard, indent=2))


if (
    best_hard["validation_ndcg10"]
    > best_random["validation_ndcg10"]
):

    winner = "hard"
    winner_result = best_hard

elif (
    best_random["validation_ndcg10"]
    > best_hard["validation_ndcg10"]
):

    winner = "random"
    winner_result = best_random

else:

    winner = "tie"
    winner_result = best_random


print("\n" + "=" * 75)
print(f"TASK 4 WINNER: {winner.upper()}")
print("=" * 75)


# ============================================================
# RETRAIN / LOAD BEST RANDOM MODEL
# ============================================================

print("\nPreparing final RANDOM model...")

best_random_model = SentenceTransformer(
    best_random["model_dir"]
)

# Copy best model to required final location
if os.path.exists(RANDOM_MODEL_DIR):
    shutil.rmtree(RANDOM_MODEL_DIR)

best_random_model.save(
    RANDOM_MODEL_DIR
)


# ============================================================
# RETRAIN / LOAD BEST HARD MODEL
# ============================================================

print("Preparing final HARD model...")

best_hard_model = SentenceTransformer(
    best_hard["model_dir"]
)

if os.path.exists(HARD_MODEL_DIR):
    shutil.rmtree(HARD_MODEL_DIR)

best_hard_model.save(
    HARD_MODEL_DIR
)


# ============================================================
# FINAL EVAL RETRIEVAL
# ============================================================

def generate_eval_run(
    model,
    run_name,
    output_path
):

    print(
        f"\nGenerating evaluation run: "
        f"{run_name}"
    )

    corpus_embeddings = model.encode(
        [
            doc_text[d]
            for d in all_doc_ids
        ],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    corpus_embeddings = np.asarray(
        corpus_embeddings,
        dtype=np.float32
    )

    index = faiss.IndexFlatIP(
        corpus_embeddings.shape[1]
    )

    index.add(corpus_embeddings)

    eval_embeddings = model.encode(
        [
            str(c["claim"])
            for c in eval_claims
        ],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    eval_embeddings = np.asarray(
        eval_embeddings,
        dtype=np.float32
    )

    scores, indices = index.search(
        eval_embeddings,
        100
    )

    rankings = {}

    for i, claim in enumerate(eval_claims):

        claim_id = get_claim_id(claim)

        rankings[claim_id] = [
            (
                all_doc_ids[idx],
                float(scores[i][rank])
            )
            for rank, idx in enumerate(
                indices[i]
            )
        ]

    save_trec(
        output_path,
        rankings,
        run_name
    )


# Required Task 4 files
generate_eval_run(
    best_random_model,
    "t4_random",
    os.path.join(
        PRED_DIR,
        "t4_random.trec"
    )
)

generate_eval_run(
    best_hard_model,
    "t4_hard",
    os.path.join(
        PRED_DIR,
        "t4_hard.trec"
    )
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "base_model": MODEL_NAME,
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "triplet_margin": TRIPLET_MARGIN,
    "learning_rates": LEARNING_RATES,
    "epochs_tested": EPOCHS_LIST,
    "num_random_negatives": NUM_NEGATIVES,
    "num_hard_negatives": NUM_NEGATIVES,
    "hard_negative_source": [
        "Task 2 MiniLM retrieval"
    ],
    "all_results": results,
    "best_random": best_random,
    "best_hard": best_hard,
    "winner": winner
}

summary_path = os.path.join(
    ART_DIR,
    "task4_summary.json"
)

with open(summary_path, "w") as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("TASK 4 COMPLETE")
print("=" * 75)

print(
    f"Best random nDCG@10: "
    f"{best_random['validation_ndcg10']:.4f}"
)

print(
    f"Best hard nDCG@10:   "
    f"{best_hard['validation_ndcg10']:.4f}"
)

print(
    f"Winner: {winner.upper()}"
)

print("\nGenerated:")
print("  predictions/t4_random.trec")
print("  predictions/t4_hard.trec")
print("  artifacts/task4_summary.json")
print("  artifacts/task4_random_model/")
print("  artifacts/task4_hard_model/")
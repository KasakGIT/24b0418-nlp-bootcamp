"""
Task 5: fine-tune an NLI model on train_claims.jsonl and score top docs.

Reproduce (Colab T4, ~3 min; paths below match the Colab layout):
  data/corpus.jsonl, data/train_claims.jsonl, data/val_claims.jsonl
  predictions/t3_rerank_val.trec  (val ranking; lives in artifacts/ locally)
  eval_claims.jsonl, t3_rerank.trec  (eval claims and ranking)
  pip install sentencepiece rank_bm25
  python task5_finetune_colab.py
Output: ft_scores.json -> copy to artifacts/ for task5_verify_finalize.py
Seed 42. Base model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (~184M params).
"""
import json, re, random, time, numpy as np, torch
from collections import defaultdict
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

CORPUS = "data/corpus.jsonl"
TRAIN = "data/train_claims.jsonl"
VAL = "data/val_claims.jsonl"
VAL_RANK = "predictions/t3_rerank_val.trec"
EVAL_CLAIMS = "eval_claims.jsonl"
EVAL_RANK = "t3_rerank.trec"
OUT = "ft_scores.json"

BASE = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
EPOCHS, LR, BS, MAXLEN, MAX_WIN, TOPN = 2, 1e-5, 16, 160, 2, 3
random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = "cuda"

def read_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def read_trec(p):
    runs = defaultdict(list)
    for line in open(p, encoding="utf-8"):
        if line.strip():
            x = line.split(); runs[int(x[0])].append((int(x[3]), int(x[2])))
    return {c: [d for _, d in sorted(r)] for c, r in runs.items()}
def sentences(d):
    ab = d.get("abstract", [])
    if isinstance(ab, str): ab = re.split(r"(?<=[.!?])\s+", ab)
    return [str(s).strip() for s in ab if str(s).strip()]
def windows(s):
    out = [" ".join(s[i:i+n]) for n in range(1, MAX_WIN+1) for i in range(len(s)-n+1)]
    return out or [" ".join(s)]
def toks(t): return set(w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) > 3)

corpus = {int(d["doc_id"]): d for d in read_jsonl(CORPUS)}
doc_list = list(corpus)
train = read_jsonl(TRAIN)
val = read_jsonl(VAL)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, torch_dtype=torch.float32).to(device).float()
print("param dtype:", next(model.parameters()).dtype)
idx = {}
for i, n in model.config.id2label.items():
    n = str(n).lower()
    if "entail" in n: idx["SUPPORT"] = int(i)
    elif "contrad" in n: idx["CONTRADICT"] = int(i)
    elif "neutral" in n: idx["NEUTRAL"] = int(i)
assert len(idx) == 3, model.config.id2label
print(idx)

# ---------- training data (train claims only) ----------
def hard_negs(claim, k=2, sample_docs=40):
    ct = toks(claim); cands = []
    for did in random.sample(doc_list, sample_docs):
        for s in sentences(corpus[did]):
            ov = len(ct & toks(s)) / max(1, len(ct))
            cands.append((ov, s))
    cands.sort(reverse=True)
    return [s for _, s in cands[:k]]

# NEUTRAL pairs from train claims with no evidence: top-3 BM25 (Task 1 system, k1=2.0, b=0.3)
# abstracts, their 2 best word-overlap sentences each.
tokz = lambda t: re.findall(r"[a-z0-9]+", t.lower())
bm25 = BM25Okapi([tokz(str(corpus[i].get("title", "")) + " " + " ".join(sentences(corpus[i]))) for i in doc_list], k1=2.0, b=0.3)
nei_pairs = []
for c in train:
    if c.get("evidence"): continue
    sc = bm25.get_scores(tokz(str(c["claim"])))
    top = sorted(range(len(doc_list)), key=lambda i: -sc[i])[:3]
    ct = toks(str(c["claim"]))
    for i in top:
        ss = sorted(sentences(corpus[doc_list[i]]), key=lambda s: -len(ct & toks(s)) / max(1, len(ct)))[:2]
        for s in ss: nei_pairs.append((s, str(c["claim"])))
print("extra NEUTRAL pairs from claims with no evidence:", len(nei_pairs))

data = []
for c in train:
    claim = str(c["claim"]); ev = c.get("evidence", []) or []
    used = defaultdict(set)
    for e in ev:
        d = corpus.get(int(e["doc_id"]))
        if d is None or str(e.get("label", "")).upper() not in idx: continue
        s = sentences(d)
        ids = [i for i in e.get("sentences", []) if i < len(s)]
        if not ids: continue
        used[int(e["doc_id"])].update(ids)
        data.append((" ".join(s[i] for i in ids), claim, idx[str(e["label"]).upper()]))
    for did, ids in used.items():
        s = sentences(corpus[did]); rest = [x for i, x in enumerate(s) if i not in ids]
        for x in random.sample(rest, min(2, len(rest))): data.append((x, claim, idx["NEUTRAL"]))
    for x in hard_negs(claim, 2 if not ev else 1): data.append((x, claim, idx["NEUTRAL"]))
data += [(x, cl, idx["NEUTRAL"]) for x, cl in nei_pairs]
cnt = defaultdict(int)
for _, _, y in data: cnt[y] += 1
print(len(data), "training pairs, class counts:", dict(cnt))

# ---------- train (fp32, nan-guarded) ----------
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * ((len(data) + BS - 1) // BS)
sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
model.train(); t0 = time.time()
for ep in range(EPOCHS):
    random.shuffle(data); tot, n_ok, n_skip = 0.0, 0, 0
    for s in range(0, len(data), BS):
        b = data[s:s+BS]
        enc = tok([x[0] for x in b], [x[1] for x in b], padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt").to(device)
        loss = model(**enc, labels=torch.tensor([x[2] for x in b]).to(device)).loss
        if not torch.isfinite(loss):
            opt.zero_grad(); n_skip += 1; continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad(); n_skip += 1; continue
        opt.step(); sch.step(); opt.zero_grad()
        tot += loss.item() * len(b); n_ok += len(b)
    print(f"epoch {ep+1} loss {tot/max(1,n_ok):.4f} skipped_batches {n_skip}  {time.time()-t0:.0f}s", flush=True)
model.eval()

# ---------- scoring ----------
def score(claims, rank):
    pairs, owners = [], []
    for c in claims:
        cid = int(c["id"])
        for pos, did in enumerate(rank.get(cid, [])[:TOPN]):
            if did in corpus:
                for w in windows(sentences(corpus[did])):
                    pairs.append((w, str(c["claim"]))); owners.append((cid, did, pos))
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0])); best = {}
    with torch.no_grad():
        for s in range(0, len(order), 128):
            ids = order[s:s+128]
            enc = tok([pairs[i][0] for i in ids], [pairs[i][1] for i in ids], padding=True,
                      truncation=True, max_length=MAXLEN, return_tensors="pt").to(device)
            p = torch.softmax(model(**enc).logits, -1).cpu().numpy()
            for i, row in zip(ids, p):
                ps, pc, pn = float(row[idx["SUPPORT"]]), float(row[idx["CONTRADICT"]]), float(row[idx["NEUTRAL"]])
                k = owners[i]; m = max(ps, pc) - pn
                if k not in best or m > best[k][3]: best[k] = (ps, pc, pn, m)
    return {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in best.items()}

val_s = score(val, read_trec(VAL_RANK))
ev_s = score(read_jsonl(EVAL_CLAIMS), read_trec(EVAL_RANK))
json.dump({"val": val_s, "eval": ev_s}, open(OUT, "w"))

# ---------- validation sweep (threshold and N chosen on val only) ----------
gold = set()
for c in val:
    for e in c.get("evidence", []) or []:
        if str(e.get("label", "")).upper() in ("SUPPORT", "CONTRADICT"):
            gold.add((int(c["id"]), int(e["doc_id"]), str(e["label"]).upper()))
rows = defaultdict(list)
for k, (ps, pc, pn, m) in val_s.items():
    cid, did, pos = map(int, k.split("|")); rows[cid].append((pos, did, ps, pc, pn))
for N in (1, 2, 3):
    best = (0, 0, 0, 0, 0)
    for thr in np.arange(-1.0, 1.0, 0.05):
        pred = set()
        for cid, rs in rows.items():
            c = sorted(((max(ps, pc) - pn, did, "SUPPORT" if ps >= pc else "CONTRADICT")
                        for pos, did, ps, pc, pn in rs if pos < N), reverse=True)
            if c and c[0][0] >= thr: pred.add((cid, c[0][1], c[0][2]))
        tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best[0]: best = (f1, thr, tp, fp, fn)
    print(f"FT N={N} thr={best[1]:+.2f} F1={best[0]:.4f} TP={best[2]} FP={best[3]} FN={best[4]}")

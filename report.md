# SciFact Claim Verification - Results

## Q1. BM25 vs. general dense retrieval
BM25 did better on claims 417 and 645. In both, words from the claim ("reproductive period", "vitamin B6") appear in the gold abstract's title, so exact matching finds it, while dense search ranked it 15th and 12th. Dense did better on 55 and 537: claim 55 mentions AMPK but the gold abstract is about metformin, and claim 537 says "hypocretin" where the abstract says "orexin", so BM25 missed them (537 was outside its top 100). Over all validation claims they win equally often, 20 each.

## Q2. Task 3 gain and latency
Fusion (RRF of BM25 and MiniLM) gave the biggest gain, but only just: 0.8155 (best single retriever) to 0.8418, +0.026. The cross-encoder reranker added a similar +0.026 (0.8418 to 0.8676). Fusion costs almost nothing in latency; I did not time it separately, and dense query encoding takes 8.6 ms. The reranker is slow: about 15.7 s per query on my CPU in the final setup (my first version took 2.9 s and scored 0.8645).

## Q3. Hard negatives and overfitting
Only slightly. With lr 1e-5 and 1 epoch, hard negatives scored 0.8327 and random 0.8317, which I treat as no real difference. Training longer made both worse, hard negatives more so: 2 epochs gave 0.8074 (hard) vs 0.8279 (random), and lr 2e-5 gave 0.7992 and 0.7656 for hard. I took these drops in validation nDCG@10 as overfitting. I only tracked validation scores, not training loss, so I cannot say more. A second round with 4 negatives and a lower lr did not beat 0.8327.

## Q4. Verification errors that are retrieval misses
53 of the 162 validation claims did not exactly match gold, and 8 of them (15.1%) were retrieval_miss, meaning a gold abstract was not in the reranker's top 10. So most errors are not retrieval problems. The other tags come from simple rules I wrote (for example, any number in the claim or gold abstract counts as "numerical"), so numerical (23) and other (11) are rough and should not be over-read.

## Q5. Next improvement
The verifier is DeBERTa-v3 NLI fine-tuned on the train claims, with validation F1 0.6911. The biggest gain came from adding NEUTRAL training examples built from claims with no evidence (0.6568 to 0.6911), because many false positives were on such claims. Other tries (more negatives, 3 epochs, class weights) did not help. Next I would use cross-validation over train plus validation claims, so thresholds are chosen on more than the 162 claims I tuned on and the score is less optimistic.

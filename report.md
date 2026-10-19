# SciFact Claim Verification - Results

## Q1. BM25 vs. general dense retrieval
BM25 wins on claims 417 and 645 (nDCG@10 1.0 vs 0.0): the claim wording ("reproductive period", "vitamin B6") appears almost verbatim in the gold title, while dense ranked those abstracts 15th and 12th. Dense wins on 55 and 537 (1.0 vs 0.0): claim 55 talks about AMPK but the gold abstract is about metformin and lung fibrosis, and claim 537 says "hypocretin" where the abstract says "orexin", so BM25 never retrieved it. Overall they split evenly: 20 claims each way.

## Q2. Task 3 gain and latency
Fusing BM25 and MiniLM with RRF lifted validation nDCG@10 from 0.8155 (best single retriever) to 0.8418, +0.026. The cross-encoder reranker added about the same on top, 0.8418 to 0.8676 (+0.026), so fusion was the largest gain, but only by a hair. The costs differ: fusion only needs the dense query encoding (8.6 ms) plus BM25, which I did not time separately, while the final reranker setup took about 15.7 s per query on CPU (2.9 s for my first version, 0.8645).

## Q3. Hard negatives and overfitting
Barely. At lr 1e-5 and 1 epoch hard negatives scored 0.8327 vs 0.8317 for random, a difference I would call noise. Training longer or with a higher learning rate hurt, hard negatives more than random: 2 epochs gave 0.8074 (hard) vs 0.8279 (random), and lr 2e-5 gave 0.7992 and 0.7656 for hard. I read the drop in validation nDCG@10 as overfitting, but I only tracked validation scores, not training loss. A second round with 4 negatives and a lower lr did not beat 0.8327.

## Q4. Verification errors that are retrieval misses
53 of the 162 validation claims did not match gold exactly, and 8 of them (15.1%) were tagged retrieval_miss, meaning a gold abstract was not in the reranker's top 10. So retrieval is not the main problem. The tags come from simple rules (any number in the claim or gold abstract gives "numerical"), so numerical (23) and other (11) are rough and should not be over-read.

## Q5. Next improvement
The verifier is fine-tuned DeBERTa-v3 NLI with validation F1 0.6911. Adding NEUTRAL pairs from BM25-retrieved abstracts of claims with no evidence helped (0.6568 to 0.6911; second seed 0.6565 to 0.6747). Three further tries (the same for claims with evidence, 3 epochs, class weights) stayed within about 0.01 of it, so training tweaks have plateaued on 647 train claims. Next I would use cross-validation over train plus validation claims, to train on more labelled data and to pick thresholds on out-of-fold scores instead of the 162 validation claims alone.

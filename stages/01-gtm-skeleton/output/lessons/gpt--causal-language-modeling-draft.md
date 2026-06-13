# GPT — Causal Language Modeling

## Hook It

GPT doesn't "read" your prompt the way you think it does — it generates one token at a time, conditioned only on what came before. That constraint explains every weird behavior you've seen in production, from hallucinated company names to the model contradicting itself mid-sentence.

## Ground It

Causal language modeling trains a transformer to predict the next token given only the left context (all previous tokens). This is enforced via a causal mask — a lower-triangular attention matrix that zeroes out positions to the right. Contrast with masked language modeling (BERT), which sees bidirectional context. Cover teacher forcing (training sees ground truth, inference feeds predictions autoregressively), log-likelihood as the training objective, and why perplexity measures model confidence. Explain why the model cannot "go back" — every token is final once generated.

## Build It

**Easy:** Implement a causal mask from scratch using a triangular matrix and confirm it blocks future-token attention.

**Medium:** Run next-token prediction on a short sequence using a small GPT-2 model. Print the top-k token probabilities at each position to observe the model's confidence distribution.

**Hard:** Build a greedy decoder loop that generates 20 tokens from a seed string. Then modify it to implement top-k sampling (k=50) and compare outputs across three runs with the same seed to observe stochasticity.

## Use It

Causal LM is foundational for every text generation task in GTM — personalization, outbound email drafting, ICp description synthesis. The left-to-right constraint means prompt structure is load-bearing: the model cannot revise earlier tokens, so front-loading context and constraints produces more reliable output than burying instructions at the end. GTM cluster: **Zone 4 — Engage & Convert** (AI-assisted outreach generation, email personalization pipelines). The causal mechanism explains why few-shot examples placed before the task produce different results than examples placed after.

## Ship It

Decode strategy is a production lever. Greedy decoding is deterministic but repetitive. Top-p (nucleus) sampling trades determinism for variance — critical when generating prospecting emails at scale where duplicate detection matters. Temperature controls the entropy of the distribution; in GTM pipelines, temperature 0.2–0.4 for factual extraction (company research), 0.7–1.0 for creative copy (email variants). Token budget planning: causal LM inference cost scales with sequence length squared in the attention layer, so capping output tokens in production is not optional. Log probability monitoring catches degenerate outputs — if the model's average log-prob drops sharply mid-generation, the output has likely diverged from coherent text.

## Quiz It

Three questions: (1) Identify which attention positions are zeroed by a causal mask at position 5 in a 10-token sequence. (2) Given two prompt orderings — instructions-first vs. instructions-last — predict which produces more consistent output and justify with causal LM mechanics. (3) Calculate the joint probability of a 3-token sequence given per-position log-probs, and explain why a single low-confidence token tanks the overall sequence probability.
# Attention Mechanism — The Breakthrough

## Beat 1: Hook
Sequence models had a memory problem. By token 50, the model forgot token 5. Attention solved this by letting every position look at every other position directly — no compressed hidden state required.

---

## Beat 2: Concept
Describe the mechanism: a query vector asks "what am I looking for?", key vectors answer "here's what I contain", and the dot product between them produces a weight. Multiply those weights against value vectors. That is attention. Then scale by √d_k to prevent softmax saturation. Multi-head attention runs this in parallel across sliced subspaces. Name Transformer only after the full mechanism is laid out.

---

## Beat 3: Use It
**GTM Redirect:** Attention weights are a ranking mechanism — they assign proportional importance across candidates. This is structurally identical to how Clay's waterfall enrichment ranks data providers or how intent signal scoring weights behavioral inputs across multiple sources. Foundational for Zone 2 (Signal) and Zone 3 (Enrichment).

*Exercise hook (easy):* Given a small matrix of query-key dot products, compute the softmax attention weights by hand and identify which key receives the highest weight.

*Exercise hook (medium):* Modify the provided single-head attention code to accept a "mask" that zeros out certain positions — simulate prioritizing first-party intent data over third-party.

---

## Beat 4: Code It
Implement scaled dot-product attention from scratch using only `numpy`. Feed it a sentence of token embeddings. Print the attention weight matrix to show which tokens attend to which. Then implement multi-head attention by splitting the QKV projection into parallel heads and concatenating the output.

*Exercise hook (hard):* Extend the single-head implementation to multi-head. Verify output shape matches the single-head baseline when head count = 1.

---

## Beat 5: Ship It
Attention is O(n²) in sequence length — this is the production constraint. For GTM pipelines processing thousands of account records, this limits batch size and context window. Discuss when attention is warranted (cross-referencing multiple enrichment signals) versus when simpler retrieval (cosine similarity against embeddings) is sufficient. **GTM Redirect:** In Clay waterfall logic, provider priority ordering is a manual hard-coded attention — the model-based version would learn those weights. [CITATION NEEDED — concept: learned provider ranking in enrichment waterfalls]

*Exercise hook (medium):* Benchmark the numpy attention implementation at sequence lengths 32, 128, 512, 2048. Print the runtime. Identify where the quadratic wall hits.

---

## Beat 6: Review
Three testable objectives:

1. **Compute** scaled dot-product attention weights given Q, K, V matrices and identify the highest-attention pair.
2. **Explain** why the √d_k scaling factor prevents softmax gradient saturation.
3. **Compare** attention to RNN hidden-state compression on a 100-token sequence — identify which mechanism preserves earlier token information and measure the difference.
4. **Evaluate** whether a given GTM data-processing task (e.g., ranking 50 intent signals for an account) warrants full attention or can be solved with simpler aggregation.

*Quiz material grounded in these objectives only. No generic ML trivia.*
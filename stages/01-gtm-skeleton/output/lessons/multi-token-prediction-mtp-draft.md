# Multi-Token Prediction (MTP)

## 1. Hook

Next-token prediction is the standard training objective for autoregressive LLMs. But predicting one token at a time creates a myopic signal — the model never learns to "plan ahead" during training. Multi-token prediction forces the model to predict N future tokens simultaneously at every position, changing the gradient signal entirely.

## 2. Concept

Describe the mechanism: at each position in the sequence, the model produces K independent predictions for tokens at positions t+1, t+2, ..., t+K using separate output heads that share the same backbone. Each head computes cross-entropy loss against its respective future token. The total loss is the sum (or weighted average) across all heads. During inference, only the first head (next-token) is used — but the shared backbone has been shaped by the multi-token training signal. Contrast this with next-token-only training. Explain why this improves performance on tasks requiring longer-range planning (code generation, reasoning) per Meta's 2024 paper [CITATION NEEDED — concept: Meta MTP paper "Better & Faster LLMs via Multi-token Prediction"]. Note: inference-time speculative decoding using the auxiliary heads is a separate optimization that reuses the trained heads.

## 3. Implementation

Provide a minimal PyTorch implementation of an MTP training forward pass. A shared transformer backbone feeds K linear heads. Each head predicts the token at offset +1 through +K. Compute per-head loss and return the sum. Run it on a small synthetic sequence of token IDs. Print each head's loss value and the total loss to confirm the mechanism works.

## 4. Use It

**GTM Redirect: Zone 1 — Model Selection and Evaluation.** MTP-trained models (e.g., DeepSeek-V3 which implements this approach [CITATION NEEDED — concept: DeepSeek-V3 MTP architecture details]) produce stronger performance on structured generation tasks like email composition, enrichment schema output, and multi-field extraction — all common in GTM automation. When evaluating models for a GTM stack, checking whether the model was trained with MTP is a legitimate selection criterion for tasks requiring coherent multi-step output. Foundational for Zone 1 model evaluation decisions.

## 5. Ship It

Cover the practical deployment considerations: MTP adds parameter overhead from the extra heads (though heads are small relative to the backbone). Training compute increases roughly linearly with K. The payoff is model quality, not inference speed — unless speculative decoding is implemented, in which case the auxiliary heads are used to draft multiple candidate tokens that the main head verifies. Describe the storage vs. compute tradeoff. Note that most practitioners will consume MTP-trained models rather than train them, so the ship concern is: can your inference runtime exploit the auxiliary heads for speculative decoding?

## 6. Deep Cut

The information-theoretic angle: next-token prediction with cross-entropy is equivalent to minimizing KL divergence between the model's conditional distribution and the empirical data distribution for a single step. MTP minimizes a joint objective over K steps, which encourages the backbone to learn representations that encode longer-range structure. This is related to, but distinct from, n-gram training objectives and contrastive predictive coding. Open question: the optimal K is empirically 4–8 in published work, but there is no theoretical justification for this range [CITATION NEEDED — concept: theoretical analysis of optimal MTP horizon K].

---

### Exercise Hooks

- **Easy:** Modify the provided implementation to change K from 4 to 8. Print the per-head losses and observe how later heads (predicting further into the future) produce higher loss on the same data.
- **Medium:** Implement a greedy speculative decoding loop that uses the auxiliary MTP heads to draft K tokens at once, then verifies each against the main head's top-1 prediction. Print acceptance rate.
- **Hard:** Train a small 2-layer transformer with and without MTP on a synthetic task (e.g., generating balanced parentheses or a simple formal language). Measure and compare sample efficiency — how many training steps to reach zero validation loss.
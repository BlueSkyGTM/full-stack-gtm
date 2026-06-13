# Self-Attention from Scratch

---

## Beat 1: Hook

The transformer revolution started with one operation: letting every token in a sequence "look at" every other token and decide what matters. That operation is self-attention. If you've ever used an LLM to score leads, generate outbound, or classify support tickets, self-attention is the mechanism computing those token relationships under the hood. This lesson builds it from raw matrix math.

---

## Beat 2: Concept

Describe the three-matrix abstraction (query, key, value) and why it exists — each token produces a question, a match signal, and a content payload. Walk through the attention score computation: dot-product similarity between Q and K, scaling by √d_k to prevent softmax saturation, softmax normalization, then weighted sum over V. Show the full formula: `Attention(Q,K,V) = softmax(QK^T / √d_k) · V`. Explain why self-attention is "self" — the same sequence produces all three matrices. Briefly contrast with cross-attention (different source for Q vs K,V) but note this lesson stays on self-attention. No code yet — just the mechanism on paper.

---

## Beat 3: Demo

Implement scaled dot-product self-attention from scratch using only `torch` tensor operations (no `nn.Module`, no `F.scaled_dot_product_attention`). Start with a synthetic sequence of 6 tokens, each with a 4-dimensional embedding. Generate Q, K, V via random weight matrices. Compute attention scores, apply scaling, apply softmax, multiply by V. Print every intermediate tensor shape and value so the practitioner can trace the computation step by step. Then wrap it in a function that accepts any `(seq_len, d_model)` input and returns the attended output with attention weights. Confirm output shape matches input shape.

---

## Beat 4: Use It

This is foundational for Zone 1 (Targeting & Enrichment). Every LLM-based enrichment pipeline — Clay AI formulas, custom GPT classification, agent-based lead scoring — runs self-attention over tokenized input. You won't implement self-attention in a GTM tool; you will debug why a prompt produces unexpected scores, and that debugging requires knowing that attention weights determine which tokens influence the output. When a Clay waterfall's AI enrichment step misclassifies a lead, the failure trace goes through what the model attended to. [CITATION NEEDED — concept: attention weight analysis for prompt debugging in GTM workflows]

---

## Beat 5: Ship It

Three exercise tiers:

- **Easy**: Modify the demo function to accept a `mask` parameter that zero-outs attention to padding tokens. Print the masked attention weights to confirm padding positions receive zero probability.
- **Medium**: Implement multi-head attention by splitting the d_model dimension into `h` heads, running the demo function independently per head, then concatenating and projecting back. Print shape at each stage. Verify output dimension matches input.
- **Hard**: Build a minimal transformer block: self-attention → add & layer-norm → feed-forward → add & layer-norm. Feed a batch of company description strings through a tokenizer, embedding lookup, the transformer block, and a linear classification head. Print predicted class probabilities per company.

---

## Beat 6: Evaluate

**Learning Objectives** (testable):

1. Implement scaled dot-product self-attention using only tensor operations (no library attention functions).
2. Explain why the QK dot product is scaled by √d_k and predict what happens to softmax output without scaling.
3. Compare the role of Q, K, and V matrices and describe what each represents in the attention mechanism.
4. Detect the effect of masking on attention weight distributions by inspecting output tensors.
5. Extend single-head attention to multi-head attention by partitioning the embedding dimension.

**Quiz hooks** (not full questions — grounded in objectives above):

- Q1: Given a printed unscaled attention score matrix where values exceed 20, predict the softmax output distribution and explain why this degrades gradient flow.
- Q2: Given a 4-token sequence with d_model=8 and h=2 heads, trace the shape of the attention output after concatenation but before the final linear projection.
- Q3: A token's query vector has high dot-product similarity with two other tokens' key vectors. Explain what this means about the resulting value vector for that position.
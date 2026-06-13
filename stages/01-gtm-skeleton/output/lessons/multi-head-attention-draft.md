# Multi-Head Attention

## Hook

Single-head attention computes one set of relevance scores across a sequence. Multi-head attention runs this same mechanism in parallel across multiple independent subspaces—each head learns a different relationship pattern (syntactic, positional, semantic). This is the mechanism that lets transformers track subject-verb agreement, entity coreference, and positional proximity simultaneously.

## Learn It

Decompose the mechanism: (1) linear projections of Q, K, V into `num_heads` separate subspaces of dimension `head_dim`, (2) scaled dot-product attention computed independently per head, (3) concatenation of all head outputs, (4) final linear projection back to `model_dim`. Emphasize that each head's Q, K, V weight matrices are learned independently—the model discovers what to attend to, not the engineer.

## Use It

Foundational for Zone 1 (ICP/Enrichment). Multi-head attention is the architecture behind transformer-based intent classifiers and firmographic enrichment models that score leads by attending to multiple signal types—technographic, behavioral, firmographic—in parallel. [CITATION NEEDED — concept: transformer-based lead scoring in GTM tools]

## Build It

Implement multi-head attention from scratch in pure Python/NumPy. Print per-head attention weight matrices to confirm each head produces a distinct relevance pattern on the same input sequence. Output shapes and weight distributions at every step.

Exercise hooks:
- **Easy:** Given `batch_size=2, seq_len=4, num_heads=4, head_dim=8`, compute and print the expected output shape after concatenation and after final projection.
- **Medium:** Instrument the implementation to print each head's attention matrix separately. Verify visually that heads produce different patterns on a 6-token sequence.
- **Hard:** Run single-head vs. 4-head attention on the same input. Print cosine similarity between outputs—demonstrate that multi-head produces a richer representation.

## Ship It

Production considerations: memory scales as `O(num_heads × seq_len² × head_dim)`. Trade-off between many narrow heads (more relationship types, lower precision per head) vs. few wide heads (fewer types, more capacity per head). Configuration pattern in PyTorch: `nn.MultiheadAttention(embed_dim, num_heads)` enforces `embed_dim % num_heads == 0`. Profile GPU memory at different head counts before deploying.

## Evaluate

Assessment targets: explain why heads require independent Q/K/V projections, predict output shapes given architecture hyperparameters, diagnose a broken implementation where all heads produce identical attention patterns (weight sharing bug), compare computational cost of 8 heads at dim 64 vs. 4 heads at dim 128.

---

*GTM redirect for "Use It" and "Ship It": Zone 1 — ICP/Enrichment. Multi-head attention is a foundational subcomponent of the transformer models used in intent classification and firmographic enrichment pipelines. It is not a standalone GTM tool.*
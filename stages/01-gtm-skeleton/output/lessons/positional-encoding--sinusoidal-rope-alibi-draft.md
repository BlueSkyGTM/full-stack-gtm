# Positional Encoding — Sinusoidal, RoPE, ALiBi

## Hook
Self-attention is permutation-invariant — it cannot distinguish "prospect replied to sequence" from "sequence replied to prospect." Positional encoding is the mechanism that injects order. Without it, every transformer is a bag-of-words model. This lesson covers the three encoding families you will encounter in production LLMs and why the choice between them changes what your model can do with long sequences.

## Concept
**Sinusoidal (Vaswani et al., 2017):** Fixed basis functions. Each position gets a deterministic vector of sines and cosines at different frequencies. Added to token embeddings before attention. No learned parameters. Extrapolation beyond training length degrades.

**RoPE (Su et al., 2021):** Rotates query and key vectors in 2D subspaces by an angle proportional to position. The dot product of rotated vectors naturally encodes relative distance. Used in LLaMA, Mistral, Qwen. Enables length extrapolation with scaling techniques.

**ALiBi (Press et al., 2022):** Adds no embeddings at all. Instead, subtracts a position-dependent penalty from attention scores before softmax. Heads learn different slope rates. Used in MPT, BLOOM. Strong length extrapolation by construction.

Mechanism comparison: Sinusoidal adds position *before* attention. RoPE applies position *during* Q/K computation. ALiBi penalizes attention *after* score computation.

## Demo
Python script implementing all three encodings on a toy sequence of 8 tokens with dimension 16. Prints position vectors, rotated queries, and attention bias matrices. Visual confirmation that sinusoidal values are fixed, RoPE dot products depend on relative distance, and ALiBi produces a triangular penalty matrix.

## Use It
Context window behavior in generation tasks maps directly to the positional encoding scheme. RoPE-based models (LLaMA family) can be extended with context scaling for long-form personalization emails or multi-step sequence drafts. ALiBi-based models extrapolate more gracefully, which matters when generating output longer than training sequences. When selecting a model for Zone 1 (ICP definition) or Zone 3 (sequence writing) tasks, the encoding scheme determines whether the model degrades at 2K tokens or holds to 8K+. This is foundational for model selection in GTM workflows — not a direct application, but a decision criterion.

## Ship It
**Easy:** Print sinusoidal position vectors for positions 0–7 and verify that closer positions have higher cosine similarity.

**Medium:** Implement RoPE rotation on a random query/key pair and print the dot product matrix showing that it encodes relative distance (diagonal pattern).

**Hard:** Feed a 32-token sequence through all three encoding schemes and print the resulting attention patterns. Identify which scheme produces the most uniform attention at distance 30 versus distance 2.

## Evaluate
- Compare sinusoidal, RoPE, and ALiBi on where position information enters the attention computation (before, during, after).
- Predict what happens to attention scores at positions beyond training length for each scheme.
- Given a model card stating "RoPE with context scaling factor 4," calculate the effective context window if base training was 4K tokens.
- Explain why ALiBi requires no position embedding layer in the model architecture.

---

**Learning Objectives:**
1. Implement sinusoidal positional encodings and verify their geometric distance properties.
2. Compare where Sinusoidal, RoPE, and ALiBi inject position information into the attention computation pipeline.
3. Calculate effective context window changes given encoding scheme and scaling parameters.
4. Diagnose position-related degradation in model output when sequences exceed training length.

[CITATION NEEDED — concept: specific GTM cluster mapping for positional encoding selection in production model choice]
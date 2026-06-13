# Transformer Block from Scratch

## Learning Objectives

1. Implement a single transformer encoder block as composable sublayers (attention, FFN, residual, layer norm)
2. Trace a tensor through each sublayer and print intermediate shapes to verify data flow
3. Compare pre-norm vs. post-norm architectures by running identical inputs through both
4. Diagnose which sublayer causes a dimensional mismatch given error output
5. Configure multi-head attention hyperparameters and predict their effect on output shape

---

## Beat 1: Hook

The transformer block is the repeating structural unit inside every LLM you call from Clay, OpenAI, or Anthropic. GPT-4 stacks 120 of them. BERT stacks 12. They are all the same mechanism repeated. If you can build one, you can reason about what the entire model is doing, why it fails on certain inputs, and what "attention heads" actually compute. This is not theory — this is the assembly line your GTM prompts run on.

---

## Beat 2: Concept

A transformer block has four components wired in a specific order. Present them as a data flow, not a taxonomy:

1. **Multi-head self-attention**: computes pairwise relationships between all positions in the sequence. Each head learns a different relationship pattern (syntactic, semantic, positional). Input shape `[batch, seq_len, d_model]` → output shape `[batch, seq_len, d_model]`.
2. **Residual connection**: adds the sublayer input to the sublayer output. Purpose: gradient highway. Without this, deep stacks become untrainable.
3. **Layer normalization**: normalizes across the feature dimension to stabilize activation magnitudes. Pre-norm (normalize before sublayer) vs. post-norm (normalize after) changes training dynamics.
4. **Feed-forward network (FFN)**: two linear projections with a nonlinearity between them. Operates on each position independently. Expands to `d_ff` (typically 4× `d_model`) then projects back. This is where the model stores knowledge, per [CITATION NEEDED — concept: FFN as key-value memory, Geva et al. 2021].

Data flow for one block (post-norm variant):
```
x → attention(x) → add(x, attention_output) → layer_norm → FFN → add(normed, FFN_output) → layer_norm → output
```

Key mechanism: residual connections mean each block only needs to learn a *delta* from its input. The original signal propagates unchanged through the stack.

---

## Beat 3: Use It

**GTM Redirect**: This is foundational for Zone 1 (Signal Capture & Enrichment). The transformer block is what processes the text in your Clay waterfall enrichment steps — company descriptions, LinkedIn summaries, email bodies. When you write a prompt that says "extract intent signals from this prospect's recent activity," a stack of these blocks is what executes that extraction.

Specific connection: the multi-head attention mechanism is why an LLM can connect "Series B" in sentence 3 of a company description to "hiring 5 SDRs" in sentence 7 and infer growth stage. Each attention head learns to attend across different distance ranges and relationship types.

When you see an LLM "miss" a connection between two pieces of information in a long Clay enrichment input, the failure mode is in the attention pattern of these blocks — either the context window truncated the input, or the attention heads didn't learn to attend across that span.

Exercise hooks:
- **Easy**: Run a pre-built transformer block on a short text input. Print the output of each sublayer. Observe that shapes are preserved through the block.
- **Medium**: Modify the number of attention heads from 1 to 8. Print the attention weight matrices. Compare what different heads attend to.
- **Hard**: Feed two sentences where a critical inference requires connecting tokens across a 50-token gap. Show that a single transformer block fails but a 4-block stack succeeds.

---

## Beat 4: Build It

Implement the full transformer encoder block in Python using only `torch` and `torch.nn`. No abstraction skipping — each sublayer is a separate, inspectable module.

Code will:
1. Instantiate multi-head attention, layer norm, FFN as separate modules
2. Wire them with residual connections
3. Feed a known tensor through and print shapes + values at each stage
4. Confirm output shape matches input shape (the block is shape-preserving by design)

Every intermediate result prints to terminal. No browser. No notebook required.

Exercise hooks:
- **Easy**: Build and run the block. Verify output shape equals input shape.
- **Medium**: Implement pre-norm variant alongside post-norm. Feed identical input. Print L2 norm of outputs. Observe difference.
- **Hard**: Remove residual connections. Run the block. Print gradient magnitudes on a backward pass. Observe degradation.

---

## Beat 5: Ship It

Production considerations for transformer blocks in deployed models:

1. **Memory**: attention scales as O(seq_len²). A 4096-token input requires 16M attention weights per head per block. This is why production models limit context windows and why Clay's AI enrichment fields have character limits.
2. **Quantization**: FFN layers (the two linear projections) dominate parameter count. Quantizing them from FP32 to INT8 cuts memory by 50% per block. This is what "running a smaller model" actually means at the mechanical level.
3. **KV cache**: during inference, previously computed attention keys and values are cached so they aren't recomputed for each new token. This is why streaming outputs get faster after the first token.
4. **Batching**: multiple enrichment requests can be processed in parallel through the same transformer stack. The batch dimension in `[batch, seq_len, d_model]` exists for this reason.

Exercise hooks:
- **Easy**: Profile memory usage for sequence lengths [64, 256, 1024, 4096]. Print the quadratic scaling.
- **Medium**: Implement KV caching for single-block attention. Measure generation speed with and without cache.
- **Hard**: Implement 4-bit quantization on the FFN weights. Print the accuracy delta on a fixed test input.

---

## Beat 6: Evaluate It

Verification that the implementation is correct and the concepts landed:

1. **Shape invariant test**: feed tensors of different batch sizes and sequence lengths. Assert output shape always equals input shape.
2. **Residual connection test**: zero out the attention weights. Output should equal input (residual passes through unchanged).
3. **Determinism test**: run the same input twice. Assert identical outputs.
4. **Pre-norm vs. post-norm test**: implement both. Feed a tensor with large magnitude values. Observe which one produces more stable intermediate activations.

Exercise hooks:
- **Easy**: Write assertions for the shape invariant. Run them. Print pass/fail.
- **Medium**: Implement the zero-attention residual test. Explain why output ≠ input when layer norm has learned parameters (bias/gain).
- **Hard**: Construct an input that causes numerical overflow in post-norm but not pre-norm. Print the max activations at each sublayer to demonstrate.

---

## GTM Redirect Rules Summary

- **Primary cluster**: Zone 1 (Signal Capture & Enrichment) — transformer blocks execute the text processing in Clay enrichment steps, intent signal extraction, and entity resolution
- **Secondary cluster**: Zone 3 (Engagement & Outreach) — the same blocks generate the personalized email copy and subject lines in outbound sequences
- **Foundational note**: this is not a direct Clay feature tutorial. It is the mechanism that makes all LLM-powered GTM tools work. Understanding it lets you debug prompt failures, context window limits, and enrichment quality issues at the architectural level.
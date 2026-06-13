# Differential Attention (V2)

## Learning Objectives

1. Implement differential attention from scratch using dual softmax operations and a configurable λ scalar
2. Compare output distributions of standard single-softmax attention versus differential attention on identical inputs
3. Evaluate the effect of the λ parameter on attention noise cancellation across different input patterns
4. Diagnose when differential attention reduces entropy in the attention map versus when it introduces instability
5. Integrate differential attention into a multi-head attention block and verify output shape consistency

---

## Beat 1: Hook

Standard softmax attention distributes probability mass across all tokens — including irrelevant ones. This creates a noise floor. In retrieval-heavy pipelines (RAG, entity matching, signal detection), that noise becomes a precision problem. Differential attention subtracts two attention distributions to cancel noise. The result is sharper focus on relevant context without manual masking. This lesson builds the mechanism from the math up.

---

## Beat 2: Mechanism

**Pattern:** Split the query and key projections into two pairs. Compute two independent softmax attention maps over the same values. Subtract the second from the first, scaled by a learned scalar λ. The intuition: both softmax maps contain the same structural noise from the softmax normalization — subtraction cancels it, leaving the signal differential. The λ term is head-specific and initialized to encourage near-zero output at training start, then learned.

Key variables:
- Q1, K1, Q2, K2: split projections from the same input
- λ: per-head scalar, initialized per the paper's formula (0.8 / sqrt(2 * n_layers) or similar)
- Output: (softmax(Q1·K1^T / √d) - λ · softmax(Q2·K2^T / √d)) · V

This replaces the single softmax in standard attention. Everything else (multi-head grouping, residual connections, layer norm) stays the same.

---

## Beat 3: Code

Build differential attention in pure PyTorch. No framework abstractions. Single-file script that:

1. Generates a synthetic sequence with clear signal tokens and noise tokens
2. Runs standard attention and differential attention on the same input
3. Prints entropy of the attention map for both — differential should show lower entropy (sharper focus)
4. Varies λ and prints attention map statistics to show the parameter's effect

Output is observable: entropy values, attention weight distributions, comparison table.

Exercise hooks:
- **Easy:** Run the script as-is. Identify which tokens receive the highest attention weight differential.
- **Medium:** Modify the input sequence to add adversarial noise tokens. Observe whether differential attention maintains lower entropy.
- **Hard:** Replace the fixed λ with a per-layer learned parameter. Train for 20 steps on a synthetic task and print the learned λ values.

---

## Beat 4: Use It

**GTM Redirect:** Differential attention's noise-cancellation mechanism maps to **signal extraction in enrichment waterfalls**. When a Clay waterfall pulls data from multiple providers for the same field, the signal (correct company name, verified email) is often buried in provider noise (stale data, entity mismatches). The differential pattern — compute two distributions, subtract to cancel shared noise — is the same logic used when scoring and reconciling enrichment results across sources. This is the mechanism behind confidence-weighted waterfall outputs in [CITATION NEEDED — concept: Clay enrichment confidence scoring and multi-provider reconciliation].

**Application:** Build a minimal reconciliation scorer that takes two enrichment provider outputs for the same field, converts each to a confidence distribution, and applies the differential pattern to identify the high-confidence signal.

Exercise hooks:
- **Easy:** Implement the confidence scorer with fixed weights. Run on two mock provider outputs.
- **Medium:** Add a λ-like scalar that adjusts how aggressively the scorer cancels low-confidence overlap. Print results for three λ values.
- **Hard:** Feed the scorer 10 records with known ground truth. Print precision/recall at each λ setting.

---

## Beat 5: Ship It

Production considerations for differential attention in deployed systems:

- **Memory:** Two softmax computations per head means ~2x the attention matrix memory. For long sequences, this matters. Profile before deploying.
- **λ initialization:** The paper specifies a particular init scheme. Getting this wrong causes training instability (attention output collapses to zero). Print λ values during the first 100 steps to verify they're moving.
- **Inference cost:** At inference, the two softmax maps are independent — they can be computed in parallel. Implement as two batched matmuls, not a loop.
- **Numerical stability:** Subtracting two softmax distributions can produce negative attention weights. This is by design, but downstream layers need to handle signed values. Check for NaN after the subtraction step.
- **Monitoring:** Log per-head λ values and attention entropy. If λ → 0, the second head is being ignored and you've reverted to standard attention. If λ → 1, both heads contribute equally and noise cancellation is maximized.

Exercise hooks:
- **Easy:** Add logging to the code from Beat 3 that prints per-head λ values and entropy every 5 steps.
- **Medium:** Profile memory usage for sequence lengths [128, 512, 2048] with standard vs. differential attention. Print a comparison table.
- **Hard:** Implement gradient checkpointing for the differential attention block. Measure and print the memory/performance tradeoff.

---

## Beat 6: Extend

Paths for continued work:

- **Causal masking:** This lesson used bidirectional attention. Add causal masking to the differential mechanism and verify it still reduces entropy on autoregressive tasks.
- **Flash Attention compatibility:** The dual-softmax pattern doesn't map cleanly to standard Flash Attention kernels. Research the Differential Transformer's approach to fused kernels [CITATION NEEDED — concept: Differential Transformer fused kernel implementation].
- **Sparse variant:** Instead of two dense softmax maps, explore sparse top-k attention for one or both maps. Measure the speed/quality tradeoff.
- **Multi-head λ:** The paper uses per-head λ. Implement per-layer and per-head λ, then compare training dynamics.
- **Downstream task evaluation:** Train a small classification model with standard attention, then swap in differential attention. Compare accuracy on a noisy-label dataset where the noise-cancellation property should help.

Exercise hooks:
- **Easy:** Add causal masking to the Beat 3 code. Print attention maps to verify the mask is applied.
- **Medium:** Implement per-layer λ (shared across heads in a layer) and compare learned values against per-head λ on the same training run.
- **Hard:** Build a small classifier (1-2 layers) that can toggle between standard and differential attention. Train on a synthetic noisy-label task. Print accuracy comparison.

---

## Docs Sync

Objectives written to this spec can be tested via:
- Code inspection (did they implement the dual-softmax subtraction correctly?)
- Output interpretation (does the entropy comparison show expected behavior?)
- Parameter tuning (can they explain what λ does based on observed output?)
- No trivia — every assessment references the mechanism as implemented
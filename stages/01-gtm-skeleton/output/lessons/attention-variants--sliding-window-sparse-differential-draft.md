# Attention Variants — Sliding Window, Sparse, Differential

---

## Beat 1: Hook

Standard multi-head attention scales quadratically with sequence length. Every variant in this lesson exists because someone hit that wall and needed a different complexity-accuracy tradeoff. You'll hit the same wall the first time you try to run inference on a 128K-token company research corpus and watch your compute budget vaporize.

---

## Beat 2: Concept

Three strategies for reducing the full attention matrix to something tractable. Sliding window restricts each token to a local neighborhood — O(n·w) instead of O(n²). Sparse attention selects specific token pairs (local + global + random) and zeros out everything else. Differential attention runs two softmax passes and subtracts one from the other to cancel attention noise. Each mechanism trades a different resource: window trades long-range dependencies for speed, sparse trades density for selectivity, differential trades head count for signal-to-noise ratio.

---

## Beat 3: Mechanism

**Sliding window attention** — each query token attends only to tokens within positions `[i-w, i+w]`. The attention mask is a diagonal band matrix. Implemented as a mask applied before softmax. Mistral 7B uses this with w=4096. Information propagates beyond the window through stacked layers — token at position 0 reaches position `w × layers` at the output.

**Sparse attention** — the attention pattern is defined by a fixed sparsity mask. Longformer combines local windows with a few "global" tokens that attend to everything (e.g., [CLS] token). BigBird adds random connections on top. The mask is deterministic at inference time. Complexity drops to O(n·g + n·w) where g is the number of global tokens.

**Differential attention** — two separate query-key pairs per head produce two attention distributions. The output is `softmax(Q₁K₁ᵀ)V - λ·softmax(Q₂K₂ᵀ)V` where λ is a learnable scalar. The subtraction cancels irrelevant attention mass. Introduced in [CITATION NEEDED — concept: Differential Attention paper, Microsoft Research 2024]. The mechanism is analogous to noise-canceling: two signals sharing the same noise floor, subtracted to leave the signal.

Exercise hooks:
- **Easy**: Implement a diagonal band mask for sliding window attention and print the mask shape and sparsity ratio.
- **Medium**: Build a sparse attention mask combining local windows with 2 global tokens, verify the resulting attention pattern prints correct nonzero counts.
- **Hard**: Implement differential attention with two Q-K projections and a learnable λ, run it on a short sequence, and print the difference in entropy between standard and differential attention distributions.

---

## Beat 4: Use It

**GTM Redirect**: Zone 2 — Enrichment pipelines that process long-form company data (10-K filings, earnings transcripts, multi-page technical docs).

When you enrich a company profile by feeding its full 10-K into a model, the attention variant determines what the model "sees." A sliding-window model processes the filing in local chunks — it will catch section-level patterns but may miss cross-document references. A sparse-attention model with global tokens at section headers can connect the CEO letter to the risk factors section. Differential attention reduces the chance that boilerplate legal text drowns out the strategic signal.

Concrete application: if you're using a long-context model via API to summarize prospect research documents, the attention variant affects whether the summary captures cross-page themes or just locally prominent phrases. [CITATION NEEDED — concept: specific API models and their attention variants for production use].

Exercise hooks:
- **Easy**: Feed a 4000-token synthetic "company filing" through a sliding-window attention implementation and print which token pairs received nonzero attention.
- **Medium**: Compare outputs from sliding-window vs. full attention on a text with a key reference in paragraph 1 and its explanation in paragraph 20. Print both results side by side.

---

## Beat 5: Ship It

**GTM Redirect**: Zone 2 — Enrichment pipeline infrastructure. The attention variant is an infrastructure choice, not a user-facing feature.

Shipping a model with a non-standard attention variant means you need to: (1) verify the model's tokenizer and attention implementation match during export, (2) test on your actual input length distribution, not benchmarks, and (3) profile memory vs. accuracy on your specific enrichment task. Sliding window models serialize smaller and run faster on long sequences — this matters if you're processing thousands of accounts. Sparse attention requires the sparsity pattern to be compatible with your inference backend (not all ONNX/TensorRT builds support arbitrary sparse masks). Differential attention doubles the Q-K projection parameters per head — check that your throughput target survives the overhead.

Exercise hooks:
- **Easy**: Profile peak memory usage of sliding-window vs. full attention at sequence lengths [512, 2048, 8192]. Print a comparison table.
- **Medium**: Implement a sparse attention pattern for a real enrichment task (extract company names from a synthetic 2000-token document), measure latency, and print the result with timing.
- **Hard**: Build a benchmark harness that runs the same extraction task across all three attention variants and prints a latency/accuracy/memory table.

---

## Beat 6: Evaluate

Quiz questions grounded in the mechanisms above. Each question targets a specific learning objective.

**Learning objectives** (3–5, action verbs only):
1. Implement a sliding window attention mask and compute its sparsity ratio for a given window size and sequence length.
2. Construct a sparse attention pattern that combines local windows with global token positions.
3. Compare the computational complexity of sliding window, sparse, and full attention for a given sequence length.
4. Implement differential attention with dual softmax and a learnable scaling parameter.
5. Evaluate which attention variant is appropriate for a given sequence-length and accuracy requirement.

Exercise hooks:
- **Easy**: Given sequence length n=1024 and window size w=256, calculate the number of nonzero entries in the sliding window attention mask vs. full attention. Print both.
- **Medium**: Explain in one printed statement why differential attention reduces attention-to-noise tokens, referencing the subtraction mechanism.
- **Hard**: Given a production requirement (process 10K tokens, <500ms latency, must connect token 1 to token 9000), determine which attention variant fits and print the justification with complexity math.
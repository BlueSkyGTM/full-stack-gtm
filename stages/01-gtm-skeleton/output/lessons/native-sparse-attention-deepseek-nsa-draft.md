# Native Sparse Attention (DeepSeek NSA)

## Beat 1: Hook

DeepSeek-V3 and related models process 128K-token contexts without the quadratic memory blowup that standard attention would require. NSA is the mechanism that makes this possible — a hardware-aligned sparse attention pattern that routes tokens through compression, selection, and sliding window branches in parallel.

## Beat 2: Concept

Describe the three-branch NSA architecture (compression for coarse-grained token blocks, selection for salient block identification via gating, sliding window for local precision). Explain why native sparsity differs from post-hoc sparsification: the model trains with the sparse mask from step one, so it learns to route information rather than lose it. Cover the hardware alignment argument — parallel branch execution mapped to tensor cores avoids the memory-bound bottleneck of sequential sparse lookups. [CITATION NEEDED — concept: DeepSeek NSA gating mechanism specifics and training objective details from original paper]

## Beat 3: Code

Build a minimal three-branch sparse attention simulator in Python that takes a synthetic sequence, applies block compression (mean-pool or max-pool over blocks), computes a top-k block selection gate, adds a local sliding window mask, and prints the effective attention density (active tokens / total tokens) alongside the dense baseline. All output printed to terminal — no GPU required.

## Beat 4: Use It

In GTM workflows processing long-form company research, earnings transcripts, or multi-page technical RFPs, context length directly determines whether the model can ground its output in the full document or must chunk and lose cross-section references. NSA-style sparse patterns explain why certain models handle 50+ page inputs coherently while others degrade. This maps to **Zone 1 (Intelligence Foundation)** — model architecture literacy for practitioners evaluating which models can handle long-context enrichment tasks like account research compilation and ICP scoring from multi-source firmographic dumps.

## Beat 5: Ship It

**Easy:** Modify the sparse attention simulator to accept a configurable block size and observe how density changes. **Medium:** Implement the gating score as a learnable linear layer (random init, single forward pass) instead of hardcoded top-k, print the gate weights and selected block indices. **Hard:** Feed a real long document (e.g., a 10-K filing via SEC EDGAR plaintext), chunk it into blocks, run the three-branch selector, and print which document sections receive the highest selection scores — then compare those sections to what a dense model summary identifies as key passages.

## Beat 6: Assess

Quiz questions grounded in the mechanism: (1) why does native sparsity during training outperform post-hoc sparse masking at inference, (2) which branch handles local token dependencies versus global information routing, (3) given a block size of 64 and a selection budget of 8 blocks from a 2048-token sequence, compute the effective attention density and compare to a sliding window of 256 tokens. No trivia — every question maps to a learning objective.

---

**GTM Redirect:** Zone 1 (Intelligence Foundation) — foundational model architecture literacy. No forced application; NSA is an under-the-hood mechanism that practitioners encounter when choosing between long-context models for account research and document-heavy enrichment.
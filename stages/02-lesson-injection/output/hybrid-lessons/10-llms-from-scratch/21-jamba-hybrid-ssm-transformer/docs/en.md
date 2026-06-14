# Lesson: Jamba — Hybrid SSM-Transformer

## Learning Objectives

- Explain the three primitives in a Jamba block — Transformer attention layers, selective SSM (Mamba) layers, and MoE MLP — and why the 1:7 interleaving ratio exists.
- State what an SSM's recurrence looks like at the equation level and why it produces constant-memory inference regardless of sequence length.
- Compute the KV cache footprint of a Jamba-class model versus a pure-Transformer model at 256K context and quantify the difference.
- Name the Mamba-3 innovations (exponential-trapezoidal discretization, complex-valued state update, MIMO projections) and the specific weakness each targets.
- Compare linear versus quadratic scaling empirically by measuring latency across increasing context lengths.

## The Problem

Attention is quadratic in sequence length. State space models are linear. That difference is not academic — it shows up as a concrete memory bill every time you try to process a long GTM artifact. A 45-minute sales call transcript is roughly 8,000 tokens. A full day of multi-agent enrichment traces with scraped web pages, CRM notes, and email threads can hit 100,000+ tokens. At 256K tokens, a pure-Transformer attention map is 65 billion entries per head. An SSM's recurrent state is the same fixed size whether you feed it 256 tokens or 256,000.

Pure-SSM models (Mamba, Mamba-2) match Transformer perplexity at small scales but degrade on specific task categories: state tracking, in-context retrieval, and copy operations. The intuition is mechanical — SSMs compress the entire history into a fixed-size state vector. When history gets long or requires precise recall of a specific token at a specific position, information leaks out of that compression. Attention has no such limitation because it never compresses; every token's key-value pair sits in memory, available for direct lookup. The cost is that you pay O(n²) compute and O(n) memory for the privilege.

Neither architecture alone handles what GTM engineering actually demands. A multi-agent system that processes call transcripts, enrichment payloads, and CRM history produces traces that are long *and* require precise retrieval. You need the SSM's ability to carry 256K tokens of context without crashing your GPU, and you need attention's ability to find the one sentence in a transcript where the prospect mentioned a competitor. Jamba's answer is to stop choosing and interleave both.

## The Concept

Jamba stacks two sequence-modeling primitives in a single architecture. The first is the selective state space model — specifically, the Mamba variant. An SSM maintains a recurrent state $h_t$ that updates as it reads each token: $h_t = \bar{A} h_{t-1} + \bar{B} x_t$, where $\bar{A}$ and $\bar{B}$ are discretized state transition matrices and $x_t$ is the input at position $t$. Mamba's innovation (over earlier SSMs like S4) is making $\bar{B}$ and another matrix input-dependent — the model learns *what to remember* based on the current token, not just *how to transition*. This selective mechanism runs in O(n) time because each step is a fixed-size matrix multiply, and the state $h_t$ is the same dimension whether you are at token 10 or token 250,000.

The second primitive is standard Transformer multi-head attention. Attention computes scaled dot-product similarities between every pair of positions: $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$. This requires materializing an $n \times n$ similarity matrix
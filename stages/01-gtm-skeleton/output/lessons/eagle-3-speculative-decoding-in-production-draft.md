# EAGLE-3 Speculative Decoding in Production

## Hook It
Speculative decoding trades compute for latency: a draft model proposes tokens, the target model verifies them in a single forward pass. EAGLE-3 accelerates this by drafting from the target model's own feature-space rather than a separate small model, yielding higher acceptance rates and tree-structured candidate sequences.

## Map It
Covers the mechanism: how draft tokens are generated from top-layer hidden states, how tree-structured speculation expands candidate branches, how the target model's verification pass accepts or rejects each branch, and why feature-level drafting outperforms token-level drafting from a separate model. Positions EAGLE-3 relative to vanilla speculative decoding, Medusa, and lookahead decoding.

## Build It
Implements a minimal speculative decoding loop that demonstrates the accept/reject mechanism, a feature-conditioned draft stub, and a comparison of tokens-per-second with and without speculation. Code runs a toy model pair through the loop and prints acceptance rate, wall-clock time, and throughput delta.

- **Easy**: Run the provided loop on two model sizes, report acceptance rate and speedup ratio.
- **Medium**: Modify the draft branching factor from linear to tree (width=3, depth=2) and measure the throughput change.
- **Hard**: Profile GPU memory overhead as candidate tree width scales from 2 to 16; identify the saturation point where memory costs negate latency gains.

## Use It
Speculative decoding is foundational for any GTM workflow that runs batch LLM inference at scale — high-volume personalized email generation, bulk ICP scoring, or automated research summarization. When inference throughput is the bottleneck between "pilot" and "production," EAGLE-3-style speculation is the lever. GTM redirect: **Zone 3 Enrich** and **Zone 4 Engage**, specifically where Clay or similar platforms run many parallel inference calls for enrichment and outreach personalization.

## Ship It
Production deployment concerns: tree-attention kernel selection (FlashAttention compatibility), KV-cache prefill strategy for draft tokens, acceptance-rate monitoring as a health metric, fallback behavior when acceptance rate drops below threshold, and cost modeling (speculation trades latency for additional FLOPs — quantify the break-even point for your serving hardware).

- **Easy**: Deploy a speculative-decoding-enabled endpoint (vLLM with speculative config), send 100 requests, log acceptance rate per request.
- **Medium**: Build a Grafana/dashboard panel that tracks acceptance rate, tokens-per-second, and draft-tree depth; set an alert when acceptance rate drops below 0.6.
- **Hard**: Run a load test comparing EAGLE-3 vs. vanilla autoregressive serving at sustained 50 RPS; produce a cost-per-token analysis including the additional compute overhead of rejected drafts.

## Quiz It
Assessment hooks mapped to objectives: identify the acceptance condition in speculative decoding (verify mechanism), explain why feature-level drafting produces higher acceptance than token-level drafting, calculate effective throughput given acceptance rate and tree geometry, compare EAGLE-3's approach to Medusa's parallel head approach, and diagnose a production scenario where acceptance rate degrades (input distribution shift).
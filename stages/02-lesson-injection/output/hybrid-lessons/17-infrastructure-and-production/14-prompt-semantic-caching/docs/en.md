# Prompt Caching and Semantic Caching Economics

> **Pricing snapshot dated 2026-04.** Numeric claims reflect vendor rate cards captured at this lesson's publication; verify against the linked docs before quoting them downstream.

## Learning Objectives

- Compute the break-even point for L2 prompt caching given provider pricing, TTL selection, and expected cache read count.
- Implement an L1 semantic cache using embedding similarity and measure its hit rate, false positive rate, and net cost against a labeled workload.
- Compare total cost across no-cache, prompt-cache, and semantic-cache strategies for a simulated enrichment waterfall workload.
- Diagnose caching failure modes including prefix invalidation, TTL expiry before amortization, and semantic false positives that propagate wrong answers downstream.
- Configure an enrichment prompt template that maximizes the cacheable prefix while isolating dynamic per-record data outside the cached region.

## The Problem

Every LLM call reprocesses every token you send. If you run 1,000 enrichment calls that each share a 2,000-token system prompt, you pay for 2 million tokens of identical inference work — the model recomputes the same attention patterns from scratch each time. The KV tensors (the key-value intermediate representations computed during attention) are identical across all 1,000 calls, but the provider throws them away after each response and charges you full price to recompute them on the next one.

Provider-side prompt caching eliminates this redundancy. The provider stores the KV tensors from the first call's prefix and reuses them on subsequent calls whose prefix bytes match exactly. Cache reads cost roughly 10% of fresh input pricing on both Anthropic and OpenAI. But writing into cache carries a premium — 1.25× the base input rate for a 5-minute TTL on Anthropic, and roughly 2× for the 1-hour TTL option. You are paying upfront for the right to cheaper reads later.

The catch is that caching is a bet, not a guarantee. If your cache entry expires before anyone reads it, you paid the write premium with zero amortization — more expensive than no caching at all. If you run calls in parallel (all issued before the first cache write lands), none of them hit the cache and you pay full price on every call. If dynamic text — a timestamp, a record ID, a session token — appears inside the prefix, the hash changes on every call and the cache never fires. ProjectDiscovery reported moving from 7% to 74% prompt-cache hit rate by relocating dynamic text outside the cacheable prefix (ProjectDiscovery, 2025-11). The cache was always available; the prompt structure was defeating it.

Semantic caching operates on a different axis entirely. Instead of exact prefix matching at the provider, you embed the full query, compare its vector against stored vectors from prior queries, and serve the cached response if cosine similarity exceeds a threshold. You skip the LLM call on hits — but you pay embedding cost and vector search cost on every query regardless of outcome, and a loose threshold serves wrong answers that propagate through your pipeline.

The question is never "should I cache?" It is: given my workload's query distribution, my provider's pricing, and my tolerance for approximate answers, which caching regime pays off and where does it lose money?

## The Concept

Two caching regimes exist, operating at different layers of the stack with different cost structures and different failure modes.

**L2 prompt caching (provider-level, exact prefix match).** The LLM provider computes a hash over your prompt
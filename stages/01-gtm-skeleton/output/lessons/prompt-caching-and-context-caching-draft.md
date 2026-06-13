# Prompt Caching and Context Caching

## Anchor It
Every LLM API call re-processes the entire prompt. When your GTM pipeline sends 10,000 enrichment requests that share the same 2,000-token system prompt, you pay to tokenize and encode that prefix 10,000 times. Prompt caching is the mechanism that avoids this redundancy. This lesson covers how providers detect reusable prompt prefixes, how cache hit/miss pricing differs, and how to structure your calls to exploit it.

## Map It
Describe the mechanism: LLM inference builds a KV (key-value) cache during forward passes. When a new request arrives with an identical token prefix to a previous request, the provider can reuse the cached KV states instead of recomputing them. Cover the distinction between automatic prompt caching (Anthropic, Google) and explicit context caching (Google's `cachedContents` API, Anthropic's cache control headers). Map the variables: minimum prefix length, TTL/eviction, cache hit vs. miss pricing ratios, and how multi-turn conversations affect cacheability. Note that caching operates at the token level — semantic similarity is irrelevant; exact prefix match is the rule.

## Build It
Two working code examples. First: an Anthropic API call using `cache_control` headers with a long system prompt, demonstrating cache creation on first call and cache hit on second call (observable via `cache_creation_input_tokens` and `cache_read_input_tokens` in the response). Second: a Google Gemini `cachedContents` creation call that explicitly stores a system context, then references it in a subsequent generation call. Both examples print token counts and cache metrics to terminal.

Exercise hooks:
- **Easy**: Modify the system prompt length in the Anthropic example and observe when cache behavior triggers (threshold detection).
- **Medium**: Build a loop that sends the same cached prompt 5 times and prints cumulative cost savings based on cache hit pricing.
- **Hard**: Implement a prompt prefix manager that automatically breaks a multi-part prompt into cacheable and dynamic segments, then measures cache hit rate across 20 varied requests.

## Use It
In GTM enrichment pipelines — specifically Clay waterfalls and bulk AI enrichment workflows — you often send thousands of requests with identical system instructions (e.g., "Extract company intelligence from this web page") but varying user content. Prompt caching reduces cost by 90% and latency by ~50% on the cached prefix portion. Cover how to structure enrichment prompts: static system + few-shot examples as the cacheable prefix, dynamic company/contact data as the uncached suffix. Connect to Zone 2 (Enrich) workflows where this pattern is most prevalent. If using Clay, note that Clay's AI enrichment column may or may not expose cache controls — this behavior is not consistently documented; [CITATION NEEDED — concept: Clay AI column prompt caching behavior].

## Ship It
Production considerations: cache TTLs expire (Anthropic: 5 minutes with rolling extension on hit; Google: configurable). High-volume pipelines must account for cache cold starts and eviction patterns. Monitor `cache_read_input_tokens` vs. `total_input_tokens` as a hit-rate KPI. Budgeting: model your expected cache hit rate before committing to a pricing tier that assumes high cacheability. Multi-tenant systems share cache scopes — one tenant's prefix does not benefit another unless they share exact tokens. Cover the diagnostic pattern: log cache metrics per request, aggregate hit rate per prompt template, alert if hit rate drops below threshold (indicating prompt drift).

## Break It
Failure modes to cover: cache stampede when TTLs expire simultaneously across parallel requests; prompt changes that invalidate cache silently (even a single whitespace difference breaks the prefix match); minimum token thresholds that prevent caching of short system prompts; cost overestimation if you assume 100% hit rate but your dynamic content varies in ways that push it above the prefix boundary; provider-specific quirks (Anthropic requires cache breakpoints to be explicitly marked, Google's context cache has a minimum token count that may exceed short prompts). End with: prompt caching is a latency/cost optimization, not a correctness feature — your pipeline must produce identical results with or without cache hits.

---

**Learning Objectives (for `docs/en.md`):**
1. Configure cache control headers on Anthropic API calls and verify cache hits via token usage metrics.
2. Create and reference explicit cached contexts using the Google Gemini `cachedContents` API.
3. Structure multi-part prompts to maximize the cacheable static prefix while isolating dynamic content in the uncached suffix.
4. Calculate cost savings from cache hit rates given provider-specific pricing differentials.
5. Diagnose cache miss causes by comparing token-level prompt differences across requests.
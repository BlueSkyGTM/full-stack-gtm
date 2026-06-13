# Caching, Rate Limiting & Cost Optimization

---

## Beat 1: Hook It

You just ran a batch enrichment on 10,000 contacts. Half were duplicates from last week's run. A quarter failed because you exceeded the provider's requests-per-minute. Your OpenAI bill has a comma in it. This lesson prevents all three — mechanically, not aspirationally.

---

## Beat 2: Ground It

Three mechanisms, each solving a distinct failure mode. **Caching** stores previous API responses keyed by input hash with a TTL — avoiding re-fetching data you already paid for. **Rate limiting** constrains request throughput using a token bucket (tokens replenish at a fixed rate, each request consumes one, requests block when the bucket is empty). **Cost optimization** is not a mechanism itself — it is the emergent property of deduplication before calls, model tiering (cheapest model that solves the task), and batch endpoints where available. Rate limiting and caching are separate concerns; conflating them causes bugs. TTL expiration is a time-based invalidation strategy; LRU (least recently used) is an eviction strategy for bounded caches — both can coexist.

---

## Beat 3: Show It

Three working code examples printed to terminal. (1) A token bucket rate limiter in Python — `time.sleep` blocks when tokens are exhausted, prints timestamps confirming the throttle. (2) A dictionary-backed cache with TTL — stores mock API responses, demonstrates a cache hit on second call, prints "HIT" vs "MISS". (3) A cost calculator that counts input tokens, multiplies by model per-1K pricing (GPT-4o-mini vs GPT-4o), prints the delta in dollars.

---

## Beat 4: Try It

**Easy:** Modify the token bucket's refill rate and observe the change in request spacing via printed timestamps.

**Medium:** Wrap an enrichment function in the TTL cache. Set TTL to 2 seconds. Call it three times with a 1-second sleep between each. Print HIT/MISS for each call — predict the pattern before running.

**Hard:** Build a composite function that deduplicates a list of domains, checks the cache for each, rate-limits the remaining lookups, and prints: total input, cache hits, API calls made, estimated cost.

---

## Beat 5: Use It

This is the Clay waterfall's underlying economics. A Clay enrichment waterfall sequences multiple data providers — but without caching, every run re-fetches every row. Without rate limiting, the waterfall's parallel lookups trigger 429s from providers like Apollo or Hunter. The deduplication-before-calling pattern is how production GTM stacks reduce per-contact enrichment cost from $0.08 to $0.02. GTM cluster: **Zone 2 — Enrichment waterfall operations**. If you run Clay tables or any batch enrichment, these three mechanisms determine whether your bill is $200 or $2,000.

---

## Beat 6: Ship It

Production checklist: (1) Log cache hit rate — if it drops below 60%, your TTL is too short or your deduplication key is wrong. (2) Expose rate limiter refill rate as an environment variable, not a hardcoded constant — providers change limits without notice. (3) Monitor per-run cost by writing token counts to a CSV after each batch; a sudden spike means either duplicate inputs or a provider-side pricing change. (4) Ship with circuit breaker logic: if 10 consecutive requests return 429 or 5xx, halt the batch and alert — don't burn credits retrying into a degraded endpoint.

---

**Learning Objectives (draft for `docs/en.md`):**

1. Implement a token bucket rate limiter that throttles requests to a configurable requests-per-minute ceiling.
2. Build a TTL-based caching layer that returns cached responses within their validity window and fetches fresh data on expiration.
3. Calculate per-contact enrichment cost by combining token counts with model pricing tiers.
4. Evaluate the cost impact of deduplication, caching, and model selection on a batch enrichment workload.
5. Configure exponential backoff with jitter for retry logic on HTTP 429 responses.
# Optimization

## Hook

Opening provocation: every API call in your GTM stack costs money and time — unoptimized enrichment pipelines burn credits on duplicate lookups, stale data, and low-value prospects. Frame optimization as a systems discipline, not a cost-cutting exercise.

## Concept

Define the three axes of optimization: **cost** (dollars per successful enrichment), **latency** (wall-clock time per record), and **yield** (percentage of records that produce actionable output). Introduce the optimization trilemma — improving one axis often degrades another — and the vocabulary for reasoning about tradeoffs: marginal cost, diminishing returns, kill depth.

## Mechanism

Walk through five concrete optimization patterns with pseudocode-level logic for each:

1. **Waterfall pruning** — stop enrichment at the first hit; do not call downstream providers. Show the before/after cost math.
2. **Deduplication gates** — hash the input (domain + signal type) and check a cache before calling any API. Show a hash-based dedup function.
3. **Batching** — group N records into a single API call where the provider supports it. Show the per-record cost curve.
4. **Selective enrichment** — score records first, enrich only those above a threshold. Show a simple scoring function.
5. **Stale-data eviction** — attach TTLs to cached results; re-enrich only when TTL has expired.

One runnable code example that demonstrates waterfall pruning with mock providers and prints cost savings.

## Use It

**GTM redirect: Cluster 14 — GTM Stack Cost Management.** Map each optimization pattern to a specific GTM tool: Clay waterfalls implement pruning natively; deduplication gates map to Clay's "skip if already enriched" conditional; selective enrichment maps to lead-scoring filters before enrichment columns. Show a Clay table schema that implements kill depth and cache checks.

Exercise hooks:
- **Easy**: Given a mock waterfall log (provided as a list of dicts), calculate total cost and identify the rows where downstream calls were unnecessary.
- **Medium**: Write a function that takes a list of domains, a mock cache (dict), and a mock API call; return enriched results while skipping cached entries and printing cost savings.
- **Hard**: Build a multi-provider waterfall that accepts a cost-per-provider table and a yield-per-provider estimate; route each record through the cheapest sufficient path and print per-record cost.

## Ship It

Build a complete cost-optimized enrichment pipeline that: (1) deduplicates input domains, (2) checks a file-based cache with TTL, (3) runs a three-provider waterfall with configurable kill depth, (4) writes results and a cost report to CSV. All code runs in the terminal with mock data; output includes total cost, records enriched, cache hit rate, and average cost per enriched record.

## Evaluate

Five quiz questions grounded in the mechanisms above — no trivia, all testable from the lesson content. Questions cover: identifying when waterfall pruning saves money, calculating marginal cost of one additional provider, recognizing the tradeoff between yield and cost, explaining why dedup must happen before the waterfall, and selecting the correct TTL eviction behavior.

---

**GTM Redirect Rules for this lesson:**
- **Use It** references: Cluster 14 (GTM Stack Cost Management), Clay waterfall pruning, Clay conditional skip, lead-score-gated enrichment, credit-per-enrichment accounting.
- **Ship It** references: "this pipeline implements the same cost controls you would apply to a Clay table running enrichment at scale" — specific mechanism, not generic value.
- If an optimization pattern does not map cleanly to a GTM tool, the redirect is: "foundational for Zone 14 — cost-aware pipeline design."
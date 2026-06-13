# Prompt Caching and Semantic Caching Economics

---

## Beat 1: Hook — The Double Charge Problem

Every repeated LLM call reprocesses identical tokens. At scale — thousands of company enrichments sharing the same system prompt — you pay for the same inference twice. But caching itself costs money. The question isn't "should you cache?" but "where does caching lose money?"

---

## Beat 2: Concept — Two Caching Regimes and Their Break-Even Math

**Prompt caching (exact prefix match):** LLM providers cache the KV pairs of a prompt prefix. Subsequent calls with the same prefix skip re-computation. You pay a reduced per-token rate on cache reads, but an upfront premium to write into cache. Mechanism: prefix hashing at the provider side. Break-even is reached at ~2–3 reads per cached prefix (varies by provider pricing). Cache entries have TTLs; infrequent access means you pay the write premium without amortizing it.

**Semantic caching (approximate match):** Embed the query, compare against stored embedding vectors, return the cached response if cosine similarity exceeds a threshold. You trade inference cost for embedding cost + vector storage cost + similarity compute. Mechanism: embedding model → vector store → similarity search → threshold gate. Break-even depends on your cache hit rate, embedding cost per query, and the cost delta between a cache hit and a full inference call.

**The economic trap:** Semantic caching with a low threshold inflates false positives (wrong answers served). With a high threshold, hit rates drop below break-even. Both regimes have failure modes that cost more than no caching at all.

---

## Beat 3: Demo — Measuring Cache Hit Costs and Break-Even Points

Exercise hooks:
- **Easy:** Make repeated calls to Claude with an identical system prompt prefix (using prompt caching headers), print input token counts and cache read tokens from the API response to observe cache hits.
- **Medium:** Build a semantic cache using an in-memory numpy vector store. Embed queries, set a cosine similarity threshold, and measure hit rate and total cost across 50 queries with known overlaps. Print the break-even analysis.
- **Hard:** Run both caching strategies against the same query workload. Print a cost comparison table showing: no-cache cost, prompt-cache cost, semantic-cache cost, and net savings/loss per strategy. Parameterize the TTL and hit rate to find where semantic caching becomes more expensive than no caching.

---

## Beat 4: Use It — Enrichment Waterfall Cost Control

[CITATION NEEDED — concept: Clay enrichment waterfall architecture and system prompt reuse across company/contact records]

In a Clay-style enrichment waterfall, hundreds or thousands of company records pass through the same research prompts — "analyze this company's technology stack", "score fit against ICP", "draft personalization email." The system prompt and instructions are identical across records; only the company data changes. This is the canonical use case for provider-side prompt caching: the shared prefix gets cached, and each record pays the reduced cache-read rate on that prefix.

Semantic caching applies when multiple leads trigger near-identical research queries — e.g., two contacts at the same company asking "what does Acme Corp do?" A semantic cache can serve the first answer to the second query without a second inference call. The economic question: is your lead volume and company overlap high enough to amortize the embedding and storage costs?

---

## Beat 5: Ship It — Production Caching: TTLs, Warming, and Invalidation

Prompt caching in production requires cache warming (sending an initial request to populate the cache before high-traffic periods) and monitoring cache miss rates. Provider TTLs vary — Anthropic's prompt cache has a 5-minute TTL that resets on each hit; if your enrichment batch takes longer than the TTL between records, the cache expires and you pay the write premium again.

Semantic caching requires deciding: embedding model choice (cost vs. quality tradeoff), vector store selection (in-memory vs. persisted), TTL per cached entry (company data goes stale), and threshold tuning per use case (ICP scoring can tolerate lower thresholds than compliance-critical outputs).

Both require observability: track cache hit rate, cost per query, and — for semantic caches — a sampling pipeline that replay-checkes cached answers against fresh inference to measure false positive rates.

Exercise hook:
- **Medium:** Write a script that simulates a batch enrichment workload of 200 records with a 5-minute provider TTL. Vary the inter-request delay and print how many cache hits you achieve at each delay. Identify the batch speed threshold where prompt caching stops paying off.

---

## Beat 6: Quiz Signals — What's Testable

- **Calculate break-even:** Given prompt cache write premium (e.g., 1.25x input price) and cache read discount (e.g., 0.1x input price), how many cache reads are needed to amortize one write?
- **Semantic threshold tradeoff:** If lowering the similarity threshold from 0.95 to 0.85 increases hit rate from 15% to 40% but raises false positive rate from 2% to 18%, at what false positive rate does semantic caching cost more than no caching (given: inference cost $0.003/query, embedding cost $0.0001/query, false positive cost $0.05/incident)?
- **TTL expiration math:** Given a 5-minute TTL that resets on hit, what is the maximum inter-request interval for a batch of N records to maintain cache continuity?
- **Strategy selection:** A GTM team enriches 50 companies/day across 8 hours. Same system prompt, different company data per record. Is prompt caching or semantic caching the better choice, and what is the expected savings?
- **Identify the failure mode:** Describe a scenario where semantic caching produces a lower net ROI than no caching at all, despite a high cache hit rate.

---

**Learning Objectives:**
1. Calculate the break-even cache hit count for provider-side prompt caching given write premiums and read discounts.
2. Implement a semantic cache with configurable similarity thresholds and measure its hit rate and false positive rate.
3. Compare total cost across no-cache, prompt-cache, and semantic-cache strategies for a given query workload.
4. Evaluate cache invalidation and TTL constraints for time-sensitive enrichment data.
5. Diagnose scenarios where caching increases cost rather than reducing it.
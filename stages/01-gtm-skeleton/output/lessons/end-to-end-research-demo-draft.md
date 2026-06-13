# End-to-End Research Demo

## Hook

You have a target account. You need to know what they do, who leads them, what they're buying, and what they care about — in under 90 seconds. This lesson wires together search, scrape, extract, and synthesize into a single pipeline you can point at any company.

## Concept

The research pipeline pattern: decompose a research question into discrete fetch-extract steps, chain them with dependency-aware orchestration, and merge results into a structured output. Covers the mechanics of sequential vs. parallel fetch strategies, extraction prompt design for consistency, and error propagation when one source fails.

**Learning Objectives:**
1. Build a multi-step research pipeline that fetches, extracts, and synthesizes data from multiple sources
2. Compare sequential vs. parallel fetch strategies and their tradeoffs in rate-limited environments
3. Design extraction prompts that produce consistent structured output across heterogeneous sources
4. Implement error handling that degrades gracefully when individual sources fail
5. Evaluate synthesized research output against a defined schema for completeness and accuracy

## Demo

Single Python script that takes a company name, searches the web for recent news + LinkedIn overview + product pages, extracts key facts into a unified schema, and prints a structured research brief. Uses `httpx` for fetching and Claude API for extraction/synthesis. Every step prints its intermediate output so you can observe the pipeline moving.

**Exercise Hooks:**
- *(Easy)* Modify the extraction schema to add a new field (e.g., "recent_funding") and run the pipeline
- *(Medium)* Add a second source for the same data point and implement conflict resolution when sources disagree
- *(Hard)* Parallelize independent fetches with `asyncio`, add a timeout per source, and measure the speedup

## Use It

**GTM Redirect — Zone 2 (Enrich): Account Research Automation**

This is the pipeline behind Clay's enrichment waterfall: fetch from source A, check if the field is populated, fall through to source B, then synthesize. Every automated account research tool — Clay, Apollo, ZoomInfo — implements this fetch-extract-merge pattern. The difference is data source contracts; the mechanism is identical.

Point this pipeline at a target account list and you have the foundation for a personalized outbound sequence. The structured output feeds directly into a template renderer for first-touch emails.

## Ship It

Moving from demo to production: add caching to avoid re-fetching the same URL within a TTL, implement retry with exponential backoff for transient failures, set hard timeouts per source to prevent one slow domain from blocking the pipeline, and log every fetch + extraction result for audit. Cover cost estimation: each research run consumes N API calls at known token rates.

**Exercise Hooks:**
- *(Easy)* Add a file-based cache that stores fetched HTML with a TTL and skips re-fetching
- *(Medium)* Implement retry logic with exponential backoff for HTTP 429 and 5xx responses
- *(Hard)* Build a cost tracker that estimates total token spend per research run and halts if it exceeds a budget threshold

## Probe It

Five questions grounded in the lesson's mechanisms:

1. Given three sources that return conflicting revenue figures, what extraction-time strategy produces the most reliable synthesized output? [CITATION NEEDED — concept: multi-source conflict resolution in LLM extraction]
2. A pipeline fetches sources A, B, and C sequentially. A and B are independent; C depends on B's output. Redesign the execution order to minimize latency. Diagram the dependency graph.
3. One source in a five-source pipeline returns a 503 and your retry exhausts its attempts. What two strategies prevent this from blocking the final synthesized output?
4. You run the same research pipeline twice against the same company and get different structured outputs. Name three causes and rank them by likelihood.
5. A research pipeline costs $0.12 per run in API tokens. You need to research 10,000 accounts. Sketch a caching strategy that reduces cost by at least 60% without sacrificing coverage of recently-changed data.
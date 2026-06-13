# Query Rewriting: HyDE, Multi-Query, and Decomposition

## Hook — The Vocabulary Gap

You embed a user's query, run cosine similarity against your vector store, and get garbage back. Not because the retrieval engine is broken — because the user's words don't resemble the document's words. Embeddings compress meaning into vectors, but a query phrased as "how do we reduce churn" and a document titled "retention strategies for SaaS" may land far apart in that space. Query rewriting is the step between the user's input and the retriever that closes this gap.

## Concept — Three Rewriting Mechanisms

Three distinct failure modes, three distinct rewrites. **HyDE** (Hypothetical Document Embeddings): generate a fake answer to the query, embed the fake answer instead of the query — the hypothesis is that a wrong answer is still closer in embedding space to real answers than the question is. **Multi-Query**: generate N rephrasings of the query, retrieve for each, union the results — expands lexical coverage without guessing which phrasing is "right." **Decomposition**: break a compound question into sub-questions, retrieve or answer each independently, then synthesize — handles multi-hop reasoning where no single document contains the full answer. Each adds latency and LLM cost. Each solves a different problem.

## Demo — Rewrite, Retrieve, Compare

Three runnable scripts. Each takes the same query against the same document corpus and prints retrieved chunks with similarity scores. First script: baseline retrieval with no rewriting. Second: HyDE — one LLM call to generate a hypothetical answer, then embed that. Third: Multi-Query — one LLM call to generate five rephrasings, retrieve for each, deduplicate, print ranked results. Observable output: you see which chunks each method surfaces and can compare recall against the baseline.

## Use It — Signal Matching in GTM Enrichment Waterfalls

[CITATION NEEDED — concept: GTM enrichment waterfall query expansion] In a Clay waterfall, you cascade through data providers looking for firmographic or intent data. The same company might be "Stripe" in one provider, "Stripe Technology" in another, and "Stripe, Inc." in a third. Multi-query rewriting applied to the company name before each provider call expands the match surface. HyDE applies when you're matching a natural-language account brief ("series B fintech in Latin America") against structured provider data — generate a hypothetical company profile, then search with that. The rewriting is the pre-retrieval step; the waterfall is the retrieval.

## Ship It — Production Trade-offs

Every rewrite costs at least one LLM call before retrieval even starts. HyDE: +1 call, ~500ms latency, improves precision on technical queries but can mislead on ambiguous ones. Multi-Query: +1 call to generate variants, then N parallel retrievals, merges results — higher recall but you pay for deduplication logic. Decomposition: +1 call to decompose, then N serial or parallel sub-retrievals, then +1 synthesis call — highest cost, handles compound questions nothing else can. Ship with observability: log which rewrite was applied, log the rewritten query, log retrieval scores before and after. A/B test rewrite vs. no-rewrite on retrieval recall and downstream answer quality. No rewrite is free. Choose based on your failure mode.

## Evaluate — Does the Rewrite Help?

Write a test harness: 20 queries, each with a known-relevant document in the corpus. Run baseline retrieval, HyDE, and Multi-Query. Measure recall@5 and recall@10 for each method across all 20 queries. Print a comparison table. The evaluation metric is not "does the answer sound better" — it's "did the correct chunk make it into the context window." If HyDE doesn't improve recall on your data, don't ship it. Query rewriting is a hypothesis about your corpus, not a universal upgrade.

---

**Exercise hooks:**

- **Easy:** Run the provided baseline and HyDE scripts on a 50-document corpus. Print the top-3 chunks for each method side by side.
- **Medium:** Implement Multi-Query retrieval with reciprocal rank fusion merging. Compare recall@5 against baseline on the same 20-query test set.
- **Hard:** Build a router that classifies incoming queries as "simple," "vocabulary-mismatch," or "compound" and applies no rewrite, HyDE, or Decomposition accordingly. Measure recall improvement and latency cost per route.
# Hybrid Retrieval with BM25 and Dense Embeddings

## Hook It
A pure semantic search for "Cassandra" returns Greek mythology. A pure keyword search for "Cassandra" misses documents that only mention "NoSQL database." Hybrid retrieval runs both paths and fuses their rankings—capturing exact lexical matches and semantic neighbors in a single pass.

## Ground It
Sparse retrieval (BM25) scores documents by term frequency and document frequency—exact token overlap with IDF weighting. Dense retrieval encodes query and documents into vectors and ranks by cosine similarity—semantic proximity even with zero shared tokens. Each fails in predictable ways; hybrid retrieval exists because their failure sets are disjoint.

## Show It
Build a minimal corpus (10–15 documents). Index it twice: once with BM25 via `rank_bm25`, once with dense vectors via `sentence-transformers`. Run the same queries against both. Then implement Reciprocal Rank Fusion (RRF) to merge the two ranked lists into one. Print side-by-side rankings to reveal where each retriever wins and where fusion recovers both.

## Do It
- **Easy:** Run the provided dual-index notebook. Identify one query where BM25 outranks dense, and one where dense outranks BM25. Print the rank positions.
- **Medium:** Replace RRF with a weighted linear combination of normalized scores. Tune the α parameter (0.0 = pure BM25, 1.0 = pure dense) and report which α maximizes recall@5 on a held-out query set.
- **Hard:** Implement a cross-encoder reranker on top of the hybrid fused list. Measure whether the additional latency of reranking the top-20 hybrid results improves precision@5 over fusion alone. Print timing and metric comparison.

## Use It
In GTM enrichment pipelines, account research requires retrieving from knowledge bases where company names and technology stack keywords demand exact match (BM25's strength) while strategic context and use-case descriptions demand semantic matching (dense's strength). Hybrid retrieval is the retrieval layer behind the account intelligence lookups mapped to Zone B of `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`. When a Clay waterfall queries your indexed account data, hybrid retrieval ensures "Snowflake" matches the company *and* "data cloud" matches adjacent concepts.

## Ship It
Running two retrievers doubles query latency and index storage. In production: pre-compute both indexes, cache fused results for high-traffic queries, and log the per-retriever contribution to clicked results so you can justify the cost. If your corpus is under 50k documents and latency budget is under 200ms, hybrid is the default. Above that scale, benchmark whether a fine-tuned dense model alone matches hybrid performance—sometimes it does, and the operational simplification wins.

---

**Learning Objectives (draft, for `docs/en.md`):**
1. Implement BM25 and dense retrieval on the same corpus and compare per-query rankings.
2. Implement Reciprocal Rank Fusion to merge sparse and dense ranked lists.
3. Evaluate hybrid retrieval recall@k against standalone BM25 and standalone dense baselines.
4. Configure fusion weighting (α parameter or RRF k constant) and justify the tuning choice with data.
5. Diagnose when hybrid retrieval is unnecessary versus when single-method retrieval fails silently.
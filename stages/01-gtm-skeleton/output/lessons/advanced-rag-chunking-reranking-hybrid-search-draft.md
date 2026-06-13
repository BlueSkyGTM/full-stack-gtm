# Advanced RAG: Chunking, Reranking, Hybrid Search

---

## Beat 1: Hook

Naive RAG fails in predictable ways: chunk boundaries split critical information, pure semantic search misses exact-match terms (part numbers, company names), and the top-k results from your vector store are often not the k most useful results. Three mechanisms fix these failure modes. This lesson covers all three.

---

## Beat 2: Concept

**Chunking strategies**: Fixed-size, recursive character splitting, and semantic chunking each make different tradeoffs between context preservation and retrieval granularity. The chunk boundary problem — where a fact lands across two chunks and neither retrieves cleanly — is the root cause behind most "RAG doesn't work" complaints.

**Hybrid search**: Dense vector retrieval captures semantic similarity but fails on exact keyword matches. Sparse retrieval (BM25) captures lexical matches but misses synonyms and paraphrases. Reciprocal Rank Fusion (RRF) merges both result sets into a single ranked list without requiring score normalization.

**Two-stage reranking**: A bi-encoder (your vector store) retrieves candidates fast. A cross-encoder scores each candidate against the query with full attention between query and document tokens, trading latency for precision. This is the retrieve-then-rerank pattern.

---

## Beat 3: Demo

Build a minimal RAG pipeline that demonstrates all three mechanisms:

1. **Chunking comparison**: Split the same document with fixed-size, recursive, and sentence-based strategies. Print chunk boundaries to show where information gets severed.
2. **Hybrid search**: Implement BM25 alongside dense retrieval, fuse with RRF, and show cases where hybrid outperforms either alone (e.g., querying a part number vs. a conceptual question).
3. **Reranking**: Take top-10 retrieved results, rerank with a cross-encoder, and show the reorder delta — which documents moved up, which moved down.

Exercise hooks:
- **Easy**: Run the provided chunking comparison on a sample document. Identify which strategy preserves a specific table's structure.
- **Medium**: Implement RRF fusion from scratch given two ranked lists. Test with a query where BM25 and dense retrieval disagree.
- **Hard**: Replace the provided cross-encoder with a prompt-based reranker (LLM as judge). Compare ranking quality and latency against the cross-encoder baseline.

---

## Beat 4: Use It

**GTM redirect**: This maps to the **Enrichment** cluster — specifically, building knowledge-retrieval systems that surface account intelligence, competitive intel, or relevant case studies during prospecting workflows.

The chunking strategy you choose determines whether a Clay waterfall can retrieve the right firmographic detail when it needs it. Hybrid search matters when your GTM queries mix conceptual intent ("companies expanding into Europe") with exact-match constraints (specific industry codes, headcount ranges). Reranking is the mechanism behind "retrieve broad, then prioritize sharp" — the same pattern used when enriching accounts against a knowledge base where precision at the top of the results directly affects data quality downstream.

**Foundational note**: If your GTM stack only calls APIs and doesn't maintain a retrieval corpus, chunking and reranking are foundational for Zone 2 (enrichment architecture) rather than something you configure today.

[CITATION NEEDED — concept: Clay waterfall enrichment pattern with RAG retrieval]

---

## Beat 5: Ship It

Production considerations for each mechanism:

- **Chunking**: Evaluate retrieval quality with your actual query distribution, not a benchmark dataset. Log which chunks get retrieved and annotate whether the chunk contained sufficient context to answer. Tune chunk size and overlap based on failure analysis.
- **Hybrid search**: BM25 requires tokenization that matches your domain (company names with punctuation, industry jargon). Calibrate the RRF `k` parameter on your data — the default may over-weight one retrieval path.
- **Reranking**: Cross-encoders add 10-100ms per document. Batch and parallelize. Set a retrieval width (top-k from stage 1) that balances recall with reranking cost. Log the rank delta to detect when your first-stage retrieval is already high-quality (reranking adds minimal value) versus when it's rescuing relevant results from positions 5-10.

Exercise hooks:
- **Easy**: Add timing instrumentation to the demo pipeline. Report p50/p95 latency for each stage.
- **Medium**: Build an evaluation harness: a set of (query, expected_chunk_id) pairs. Run retrieval with and without reranking. Compute recall@5 and MRR for both.
- **Hard**: Implement an adaptive retrieval width: if first-stage confidence scores are tightly clustered, retrieve more candidates for reranking. If scores are well-separated, skip reranking entirely. Evaluate the tradeoff.

---

## Beat 6: Quiz Ready

Assessable concepts for quiz generation:

1. **Chunking**: Predict where a specific fact will land given a chunking strategy and overlap setting. Explain why recursive character splitting preserves table structure better than fixed-size chunking.
2. **Hybrid search**: Given two ranked lists (dense and sparse), compute the RRF fusion score for a specific document. Identify queries where pure semantic retrieval fails and BM25 succeeds (and vice versa).
3. **Reranking**: Explain why a cross-encoder produces different scores than a bi-encoder for the same (query, document) pair. Predict whether reranking will help given the rank delta between first-stage results.
4. **Tradeoffs**: Given a latency budget and accuracy requirement, choose between (a) larger retrieval width + reranking, (b) hybrid search without reranking, or (c) tuned single-stage retrieval. Justify the choice.

Exercise hooks:
- **Easy**: Three recall@5 questions — one per mechanism.
- **Medium**: Given a failure case in a RAG pipeline, diagnose which mechanism (chunking, retrieval, or reranking) is the bottleneck and propose a fix.
- **Hard**: Design an end-to-end evaluation for a production RAG system that tests all three mechanisms simultaneously. Define the metrics, the test set construction process, and the acceptance criteria.
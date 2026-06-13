# Literature Retrieval

## Hook

You need to answer a question, but the answer lives across 400 PDFs, a Slack archive, and three vendor docs sites. Literature retrieval is the pipeline that turns a query into the right passages — ranked, scored, and ready for synthesis.

## Concept

Cover the three retrieval families and when each fails:

1. **Lexical retrieval** (BM25, TF-IDF) — token overlap scoring. Fast, interpretable, blind to semantics. "Revenue" ≠ "ARR" in a lexical index.
2. **Dense retrieval** (embedding cosine similarity) — meaning-based matching. Handles synonymy; fails on exact token matches that matter (part numbers, SKUs, proper nouns).
3. **Hybrid retrieval** — reciprocal rank fusion or learned score merging across both result sets. The standard production pattern.

Cover chunking as a first-class retrieval parameter: fixed-window, sentence-boundary, semantic-chunking. Chunk size and overlap directly control precision/recall trade-off.

## Demonstration

Build a minimal retrieval pipeline in Python:

- Embed a small document corpus (10–15 short docs) using `sentence-transformers`
- Index chunks with both BM25 (`rank_bm25` library) and FAISS flat index
- Run a set of test queries through each retriever separately, then through a hybrid merge
- Print ranked results with scores so the practitioner can see where lexical wins, where dense wins, and where hybrid recovers both

Observable output: query → top-k results with scores and source doc labels.

## Use It

**GTM Redirect:** Account intelligence enrichment (Zone 1 — ICP & Account Identification).

Build a retrieval index over a corpus of company 10-K filings, press releases, and product pages. Given an account name, retrieve the passages most relevant to a specific pain point or initiative. This is the retrieval layer that feeds an enrichment waterfall — the same pattern Clay implements when it pulls firmographic signals from document sources before surfacing them in a table.

Exercise hooks:
- **Easy:** Retrieve passages from a pre-built index given a natural-language account signal query.
- **Medium:** Swap chunk sizes (128 vs 512 tokens) and compare recall on a labeled query set.
- **Hard:** Implement reciprocal rank fusion merging BM25 and dense results; tune the `k` parameter to maximize recall@10.

## Ship It

Build a standalone retrieval service that:

- Accepts a directory of text/PDF files and builds a persistent hybrid index
- Exposes a CLI query interface: `python retrieve.py --query "expanding into EMEA" --top-k 5`
- Outputs ranked passages with source file, chunk index, and dual scores
- Logs every query and result set to a JSONL file for later evaluation

Exercise hooks:
- **Easy:** Wire up the document ingestion pipeline and confirm index build succeeds with printed chunk count.
- **Medium:** Add BM25 alongside dense retrieval and merge via RRF; print comparative scores.
- **Hard:** Add a metadata filter (e.g., only retrieve from docs tagged "press-release") and measure the precision delta on 20 test queries.

## Evaluate It

Assess whether the practitioner can:

1. Predict which retrieval family will win on a given query type (exact match vs. semantic).
2. Diagnose chunk-size misconfiguration from retrieval failure patterns.
3. Implement and tune hybrid score fusion.
4. Evaluate retrieval quality with recall@k and MRR on a labeled test set.

Exercise hooks:
- **Easy:** Given query–relevance judgments, compute recall@5 for a single retriever.
- **Medium:** Plot recall@k curves for BM25, dense, and hybrid on the same query set.
- **Hard:** Identify the failure mode from a set of bad retrievals (chunk too large, missing BM25 for named-entity query, embedding model mismatch) and propose the fix.

---

**GTM cluster context for "Use It" section:** Account intelligence enrichment — retrieving relevant firmographic and strategic signals from unstructured document sources to feed ICP scoring and account prioritization workflows.
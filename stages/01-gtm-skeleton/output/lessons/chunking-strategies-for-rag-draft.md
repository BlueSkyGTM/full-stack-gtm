# Chunking Strategies for RAG

## Hook It
When a retrieved chunk cuts off mid-sentence or returns an entire handbook when you needed one paragraph, the problem isn't the embedding model — it's the chunking strategy. This lesson covers the algorithms that decide where one chunk ends and the next begins, and why that boundary decision determines retrieval quality.

## Ground It
Covers five chunking mechanisms: fixed-size (character/token windows), recursive character splitting (separator hierarchies), document-structure-aware (Markdown headers, HTML tags), semantic (embedding-distance boundaries), and agentic (LLM-decided boundaries). For each: the algorithm, the tradeoff (precision vs. context completeness), and when it fails. Introduces overlap as a lossy compensation for boundary information destruction. Explains why token-based boundaries matter for downstream embedding models with hard context limits.

**Exercise hooks:**
- Easy: Given a sample document, manually predict where fixed-size chunking with 200-token windows and 20-token overlap would split. Verify against code output.
- Medium: Compare retrieval results on a 5-question set using three different chunking strategies against the same source document. Measure which chunks get returned and whether answers are extractable.

## Build It
Implement four chunking approaches in Python: (1) fixed-size with configurable overlap, (2) recursive character splitting with a separator hierarchy, (3) Markdown-header-aware splitting using structural parsing, and (4) a basic semantic chunker using sentence-level embedding cosine similarity to detect topic boundaries. All implementations produce observable output: chunk text, chunk token count, boundary locations.

**Exercise hooks:**
- Easy: Run fixed-size chunking on a provided text file. Print each chunk with its index, token count, and first/last 50 characters to confirm boundaries.
- Medium: Implement recursive splitting with a custom separator hierarchy for a new document type (e.g., log files with timestamp headers). Compare output to naive fixed-size on the same input.
- Hard: Build a semantic chunker that splits on cosine-similarity dips between consecutive sentences. Tune the similarity threshold against a held-out document where you've manually annotated ideal chunk boundaries.

## Use It
RAG pipelines power GTM knowledge bases — account research repos, competitor intelligence files, product documentation for sales enablement. The chunking strategy determines whether a rep asking "what does Competitor X charge for enterprise?" retrieves the pricing table or a 3-page company overview.

**GTM redirect:** Foundational for Zone 1 — account intelligence aggregation and enrichment workflows where RAG retrieves firmographic and technographic context from ingested documents. Specifically: chunking determines whether enrichment prompts receive precise data points or irrelevant noise. [CITATION NEEDED — concept: GTM knowledge base RAG for account research]

**Exercise hooks:**
- Easy: Chunk a competitor analysis document using two strategies. Retrieve chunks for the query "competitor pricing model." Compare which chunks surface and whether they contain the answer.
- Medium: Build a retrieval evaluator: given a set of GTM questions and a chunked knowledge base, compute precision@5 for each chunking strategy. Report which strategy wins and hypothesize why.

## Ship It
Production considerations: chunk size vs. embedding model token limit (e.g., 512 tokens for `text-embedding-3-small`), metadata propagation (source URL, page number, section header) attached to each chunk for citation in downstream prompts, storage and indexing costs scaling with chunk count, versioning when source documents update. Covers the observation that most production RAG systems default to recursive splitting with overlap because it's deterministic, fast, and produces predictable chunk counts — and when that default is insufficient.

**Exercise hooks:**
- Easy: Extend any chunker from Build It to attach metadata (source filename, chunk index, character offset) to each chunk. Print metadata alongside chunk text.
- Medium: Build a chunking pipeline that processes a directory of 20+ documents, stores chunks with metadata in a local vector store, and retrieves with source attribution for 10 test queries.
- Hard: Implement a hybrid chunker that tries semantic splitting first, falls back to recursive splitting when semantic boundaries produce chunks exceeding a token limit, and logs which strategy was used per document. Measure throughput on 100 documents.

## Extend It
Parent-child chunking (retrieve small chunks, return larger parent context). Late chunking (embed the full document first, then chunk the embedding space). Context enrichment (prepend document summary to each chunk). Multimodal chunking (tables and images as separate chunks with distinct processing). The frontier is agentic chunking where an LLM proposes boundaries — mechanism unclear, results inconsistent, worth monitoring.

**Exercise hooks:**
- Medium: Implement parent-child chunking: small chunks for retrieval, parent chunks for context delivery. Evaluate whether this improves answer completeness without degrading retrieval precision.
- Hard: Implement late chunking: embed the full document, then segment the embedding sequence. Compare retrieval quality against standard chunk-then-embed on a multi-topic document where topics span chunk boundaries.
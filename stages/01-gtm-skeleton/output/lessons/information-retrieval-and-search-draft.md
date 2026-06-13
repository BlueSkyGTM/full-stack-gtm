# Information Retrieval and Search

## Hook

You search your CRM for "companies like Stripe" and get nothing — because the name doesn't match, the industry tag is wrong, and the text field you need is buried in notes. Information retrieval is the science of getting the right document back from a query, whether that query is a keyword, a natural language question, or a vector embedding. Every enrichment waterfall, every account research agent, every "find me leads like this" workflow runs on IR under the hood.

## Concept

**Beat description:** The three retrieval families — lexical (BM25/TF-IDF), semantic (dense vector similarity), and hybrid (combining both). Scoring functions: cosine similarity, dot product, BM25's probabilistic ranking. The inverted index data structure that makes keyword search fast, and the approximate nearest neighbor (ANN) algorithms (HNSW, IVF) that make vector search tractable at scale. Trade-offs: precision vs. recall, latency vs. accuracy, sparse vs. dense representations.

## Use It

**Beat description:** GTM redirect → Zone 1/2 enrichment and account research. When a Clay waterfall enriches a company, it retrieves data from multiple sources and ranks confidence. When an account research agent searches for "enterprise SaaS companies with SOC 2 compliance," it runs a retrieval pipeline against a knowledge base. The mechanism: treat your GTM data (company descriptions, technographic signals, news) as a corpus, embed it, and retrieve by similarity instead of exact match.

## Build It

**Beat description:** Implement three retrievers in Python — a TF-IDF keyword search, a dense vector search using sentence-transformers embeddings, and a hybrid that fuses both rank lists using Reciprocal Rank Fusion (RRF). Run all three against a toy corpus of company descriptions. Print ranked results for the query "B2B payments infrastructure" so the practitioner can observe where each method agrees and disagrees.

**Exercise hooks:**
- Easy: Modify the query and observe how keyword vs. semantic rankings change.
- Medium: Implement RRF from scratch instead of using a library — fuse the two ranked lists and print the reordered results.
- Hard: Replace the toy corpus with 50 real company descriptions pulled from a CSV, benchmark recall@5 against a hand-labeled relevance set, and tune the RRF k parameter.

## Ship It

**Beat description:** Production retrieval systems face three problems: stale embeddings when source data updates, latency when the corpus grows past memory, and hallucination when the retriever returns low-relevance documents to an LLM downstream. Covered: incremental reindexing strategies, choosing between in-process vector stores (FAISS) vs. hosted (Pinecone, Weaviate) vs. Postgres pgvector, and the relevance threshold tuning that prevents garbage-in-garbage-out in RAG pipelines. The GTM redirect: your enrichment agent's retrieval step is a production IR system — treat it like one.

## Extend It

**Beat description:** Reranking models (cross-encoders) that score query-document pairs jointly after initial retrieval. Multi-vector representations (ColBERT) that preserve token-level matching. Structured metadata filtering combined with vector search. Retrieval-Augmented Generation (RAG) as the downstream consumer of the retrieval pipeline.

**Exercise hooks:**
- Easy: Add a metadata filter (e.g., industry = "fintech") to the vector search and observe the reduced result set.
- Medium: Implement a simple cross-encoder reranker using `sentence-transformers` and compare top-5 relevance before and after reranking.
- Hard: Build a minimal RAG pipeline: retrieve top-3 company descriptions, feed them as context to an LLM prompt, and generate a one-paragraph account summary. Print both the retrieved context and the generated output.
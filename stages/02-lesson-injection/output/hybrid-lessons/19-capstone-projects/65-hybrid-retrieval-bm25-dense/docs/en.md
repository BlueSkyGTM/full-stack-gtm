# Hybrid Retrieval with BM25 and Dense Embeddings

## Learning Objectives

1. Implement BM25 sparse retrieval and dense embedding retrieval on a shared corpus, then compare their per-query rankings to identify where each succeeds and fails.
2. Implement Reciprocal Rank Fusion (RRF) to merge sparse and dense ranked lists, and explain why rank-based fusion outperforms score interpolation across heterogeneous retrievers.
3. Evaluate hybrid retrieval against held-out queries using recall@k and precision@k, tuning the fusion parameters to find the operating point that balances both retrieval signals.
4. Implement a cross-encoder reranker on top of fused results and measure the latency-precision trade-off of adding a reranking stage to a production retrieval pipeline.

## The Problem

A pure lexical search for "Cassandra" returns Greek mythology and the Trojan War. A pure semantic search for "Cassandra" returns the NoSQL database but misses documents that literally contain the word `Cassandra` in a code identifier, a configuration key, or a vendor name. Both retrievers are right about different things, and both are wrong about different things. The failure sets are nearly disjoint, which is the entire reason hybrid retrieval exists.

Lexical retrieval (BM25) scores documents by exact token overlap weighted by inverse document frequency. If the query contains a literal string that appears verbatim in the corpus — a function name, a product name, a configuration key — BM25 surfaces it with high confidence. When the query is a paraphrase, a synonym, or a conceptual description that shares zero tokens with the target document, BM25 returns nothing useful. It has no semantic representation. It sees tokens, not meaning.

Dense retrieval inverts the failure mode. An embedding model encodes both query and documents into a shared vector space where semantic proximity drives ranking. A query like "how do we handle cancelled uploads" finds a document about `AbortMultipartOnFail` even though they share no tokens. But dense retrieval struggles with rare identifiers, exact string matching, and out-of-distribution vocabulary. If the query contains a specific product name or error code that the embedding model saw rarely during training, the vector representation is noisy and the ranking degrades.

A production retrieval system — whether it backs a RAG pipeline, an internal search tool, or an account intelligence lookup — cannot pick one retriever and hope the query distribution favors it. Real query streams mix literal lookups with semantic questions in unpredictable proportions. Hybrid retrieval runs both paths on every query and fuses their rankings into a single list that captures the union of what each retriever gets right.

## The Concept

BM25 computes a relevance score for each document given a query using three signals: term frequency (how often the query token appears in the document), inverse document frequency (how rare the token is across the corpus), and document length normalization (to prevent long documents from scoring higher just by having more tokens). The scoring function has two tunable parameters: `k1` controls term frequency saturation (how quickly repeated occurrences stop adding score), and `b` controls length normalization strength. Typical defaults are `k1=1.5` and `b=0.75`. BM25 is called "sparse" retrieval because it operates on a sparse vector representation — each document is a vector with a dimension per vocabulary term, and most dimensions are zero.

Dense retrieval maps both query and documents into a low-dimensional continuous space (typically 384 to 1536 dimensions) using a transformer encoder model. Ranking is cosine similarity between the query vector and each document vector. There are no tokens, no term frequencies, no IDF. The model learned during training which semantic neighborhoods matter, and that knowledge is baked into the embedding weights. Dense retrieval is called "dense" because every dimension of the vector carries a non-zero value — there is no sparsity.

The fusion step is where hybrid retrieval earns its complexity. The naive approach — normalizing BM25 scores and dense scores to the same range, then taking a weighted average — fails because the score distributions are fundamentally incomparable. BM25 scores for a short query might range from 0.5 to 12.0. Dense cosine similarities might range from 0.72 to 0.89. Min-max normalizing both to [0, 1] destroys the relative gaps that make each ranking meaningful. A document at BM25 rank 1 with score 11.8 and a document at rank 2 with score 3.2 get normalized to 1.0 and 0.13 — the gap is exaggerated. Meanwhile dense rank 1 at 0.88 and rank 2 at 0.87 normalize to 1.0 and 0.95 — the gap is compressed.

Reciprocal Rank Fusion (RRF), published by Cormack, Clarke, and Buettcher in 2009, sidesteps the score comparison problem entirely. RRF ignores scores and uses only rank positions. For each document that appears in either ranked list, RRF sums the reciprocal of a constant `k` plus the document's rank in each list:

```
RRF_score(d) = Σ_i  1 / (k + rank_i(d))
```

The constant `k` (default 60) dampens the advantage of top-ranked documents so that rank 1 and rank 2 contribute nearly the same weight. This prevents one retriever's high-confidence top pick from dominating the other retriever's ranking entirely. A document ranked #2 in both lists outscores a document ranked #1 in one list and absent from the other, which is the desired behavior when you trust neither retriever absolutely.

```mermaid
flowchart TD
    Q[User Query] --> SP[Sparse Path: Tokenize Query]
    Q --> DP[Dense Path: Embed Query]
    SP --> BM[BM25 Scoring<br/>TF × IDF × Length Norm]
    DP --> CS[Cosine Similarity<br/>q_vec · d_vec]
    BM --> RL1[Ranked List A<br/>doc_idx, bm25_score]
    CS --> RL2[Ranked List B<br/>doc_idx, cosine_sim]
    RL1 --> FUS[Reciprocal Rank Fusion<br/>score = Σ 1/(k + rank)]
    RL2 --> FUS
    FUS --> OUT[Final Hybrid Ranking]
```

RRF has three properties that make it the default fusion strategy in production systems. First, it requires no score normalization — you discard the scores and use only ranks. Second, it handles documents that appear in only one list gracefully — a document ranked #3 in BM25 but absent from dense still gets `1/63`, which may be enough to surface it if no other document has broader agreement. Third, the `k` parameter is remarkably stable across corpora and query types — the original paper found `k=60` near-optimal across every dataset they tested, and subsequent work has largely confirmed this.

## Build It

Install the two retrieval libraries:

```bash
pip install rank_bm25 sentence-transformers numpy
```

The following script builds both indexes on a shared corpus, runs five queries through each retriever independently, then fuses the results using RRF. The corpus is deliberately constructed so that some queries favor BM25 (literal identifiers, product names) and others favor dense retrieval (paraphrases, conceptual descriptions). The side-by-side output reveals exactly where each retriever wins and where fusion recovers both signals.

```python
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

corpus = [
    "Snowflake is a cloud-based data warehouse platform for large-scale analytics.",
    "Databricks provides a unified analytics platform combining data engineering and ML.",
    "MongoDB is a NoSQL document database designed for high-volume storage.",
    "Cassandra is a distributed NoSQL database optimized for write-heavy workloads.",
    "Apollo was a NASA mission that landed humans on the moon using Saturn V rockets.",
    "PostgreSQL supports advanced SQL features including JSONB and full-text search.",
    "Redis is an in-memory key-value store used for caching and real-time systems.",
    "Kafka enables real-time streaming data pipelines across distributed systems.",
    "Cassandra was a Trojan prophetess cursed to speak true prophecies no one believed.",
    "BigQuery is Google's serverless enterprise data warehouse for petabyte-scale SQL.",
    "Elasticsearch provides distributed full-text search and log analytics at scale.",
    "The data cloud strategy centralizes governance, security, and analytics access.",
    "Spanner is Google's globally distributed relational database with strong consistency.",
    "ClickHouse is a columnar OLAP database optimized for real-time analytical queries.",
    "DynamoDB is AWS's managed NoSQL key-value database for single-digit-millisecond latency.",
]

tokenized = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = model.encode(corpus, normalize_embeddings=True)

def bm25_search(query, k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    order = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in order]

def dense_search(query, k=5):
    q_vec = model.encode([query], normalize_embeddings=True)
    sims = (q_vec @ doc_vecs.T)[0]
    order = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in order]

def rrf_fuse(ranked_lists, k=60):
    scores = {}
    for lst in ranked_lists:
        for rank, (doc_idx, _) in enumerate(lst):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

queries = [
    "Cassandra NoSQL",
    "cloud data warehouse analytics",
    "prophetess Greek mythology",
    "key-value cache low latency",
    "distributed streaming pipeline",
]

for q in queries:
    bm = bm25_search(q, k=5)
    dn = dense_search(q, k=5)
    fused = rrf_fuse([bm, dn])[:5]

    print(f"\n{'='*70}")
    print(f"  QUERY: {q}")
    print(f"{'='*70}")

    print(f"\n  BM25 Top-5:")
    for r, (idx, sc) in enumerate(bm):
        print(f"    {r+1}. [{sc:7.4f}] {corpus[idx][:65]}")

    print(f"\n  Dense Top-5:")
    for r, (idx, sc) in enumerate(dn):
        print(f"    {r+1}. [{sc:.4f}] {corpus[idx][:65]}")

    print(f"\n  RRF Fused Top-5:")
    for r, (idx, sc) in enumerate(fused):
        bm_rank = next((i+1 for i,(d,_) in enumerate(bm) if d==idx), "-")
        dn_rank = next((i+1 for i,(d,_) in enumerate(dn) if d==idx), "-")
        print(f"    {r+1}. [{sc:.4f}] bm25={bm_rank} dense={dn_rank} | {corpus[idx][:45]}")
```

When you run this, watch for three patterns. First, "Cassandra NoSQL" — BM25 puts both the database and the prophetess document in its top-5 because both contain the literal token "Cassandra." Dense retrieval separates them because the database document is semantically closer to "NoSQL." RRF surfaces the database document at rank 1 because both retrievers agree on it, and pushes the prophetess document down because only BM25 voted for it.

Second, "prophetess Greek mythology" — BM25 returns nothing relevant because no document contains "prophetess" or "Greek" or "mythology." Dense retrieval finds the Cassandra prophetess document via semantic proximity. RRF surfaces it at rank 1 because the dense retriever voted for it with sufficient rank weight.

Third, "cloud data warehouse analytics" — both retrievers agree on Snowflake and BigQuery, but BM25 also surfaces the "data cloud strategy" document (high token overlap on "data" and "cloud") while dense retrieval prefers ClickHouse (semantically close to "analytics"). RRF merges both, and you get the broadest coverage of relevant documents in the fused top-5.

## Use It

Account research in a GTM enrichment pipeline is a retrieval problem with two query distributions at once. When a Clay waterfall queries your indexed account knowledge base for a specific company like "Snowflake," the retriever needs exact lexical match — the company name is a proper noun that appears verbatim in your enrichment data, case studies, and CRM notes. BM25 handles this reliably because "Snowflake" is a rare token in the corpus with high IDF weight. A dense embedding of "Snowflake" might cluster near other data infrastructure companies and return Databricks or BigQuery first, which is wrong for an exact-match account lookup.

When the same enrichment pipeline queries for strategic context — "companies migrating from on-prem data warehouses to cloud-native analytics" — there is no single literal token to match. The relevant documents might describe a customer's "modernization journey," a "shift from Teradata to Snowflake," or a "cloud-first analytics transformation." None of these share tokens with the query. Dense retrieval finds them via semantic proximity. BM25 returns documents that happen to contain the words "data" and "warehouse" regardless of strategic relevance.

Hybrid retrieval, as the retrieval layer behind account intelligence lookups mapped to Zone 19 (RAG: knowledge-augmented outreach) of the GTM topic map, handles both query types from the same endpoint without requiring the enrichment pipeline to know which retriever to call. [CITATION NEEDED — concept: Zone 19 RAG topic mapping in `stages/00-b-gtm-content-mapping/output/gtm-topic-map.md`] When a Clay waterfall searches your indexed case studies and customer stories, hybrid retrieval ensures "Snowflake" matches the exact company name (BM25 signal) while "data cloud migration use case" matches the adjacent strategic narrative (dense signal). The RRF fusion guarantees that a document about Snowflake's data cloud migration appears in the final ranking regardless of which retriever path found it first.

This matters for outbound specifically because the most effective personalized outreach references both a literal fact about the account (the company name, a technology they use, a recent hire) and a strategic narrative (why your solution fits their trajectory). Retrieving from a knowledge base that contains customer stories, case studies, and product documentation requires both signals in every query. A pure-BM25 system surfaces the right company but the wrong story. A pure-dense system surfaces the right story but sometimes the wrong company. Hybrid retrieval with RRF gets both right in a single pass.

## Ship It

Running two retrievers in production doubles query latency and roughly doubles index storage. Before you accept that cost, measure whether your corpus and query distribution actually need hybrid retrieval. The decision boundary is not about corpus size alone — it is about the diversity of query types your system receives.

If your corpus is under 50,000 documents and your latency budget is under 200ms, hybrid retrieval with BM25 and dense is the default architecture. At this scale, BM25 scoring is sub-millisecond after tokenization, dense similarity is a single matrix multiplication against a pre-computed index, and RRF fusion is negligible. The operational overhead of maintaining two indexes is real but manageable — you re-tokenize for BM25 when documents change, and you re-embed for dense when documents change. Both can be batched and run asynchronously.

Above 50,000 documents, dense retrieval alone may match hybrid performance if your query distribution is predominantly semantic — if users rarely search for exact identifiers. Benchmark this by holding out a representative query set with labeled relevance judgments and comparing recall@10 between hybrid and dense-only. If dense-only recall@10 is within 2-3% of hybrid on your actual query distribution, the operational simplification of maintaining a single index wins. If exact-match queries account for more than ~20% of your traffic, keep hybrid — the recall gap will widen on those queries.

For caching: pre-compute fused results for your highest-traffic queries. In a GTM enrichment context, the top 500 company names and technology queries account for a disproportionate share of lookups. Caching their hybrid results eliminates the dual-retriever latency for those queries entirely. Log which retriever contributed each result that a user ultimately engaged with (clicked, used in a generated email, referenced in a workflow) — this per-retriever contribution metric justifies the infrastructure cost over time and reveals when one retriever is consistently outvoted by the other on your specific query distribution.

If you add a cross-encoder reranker on top of the fused list (see the Hard exercise), budget an additional 50-150ms per query for the top-20 candidates. Cross-encoders are slow because they encode each query-document pair jointly rather than independently, but they are more accurate than either retriever alone because they see the full query and document context simultaneously. In practice, cross-encoder reranking on top of hybrid retrieval improves precision@5 by 5-15% over fusion alone, which matters when the top results feed directly into generated outbound copy where a wrong retrieval means a wrong email.

## Exercises

**Easy.** Run the Build It script above. For the query "Cassandra NoSQL," identify which document BM25 ranks at position 1 and which document dense ranks at position 1. Then check the RRF fused ranking — does fusion agree with BM25, dense, or pick a third document? Print the rank positions in a format like:

```python
query = "Cassandra NoSQL"
bm = bm25_search(query, k=5)
dn = dense_search(query, k=5)
fused = rrf_fuse([bm, dn])[:5]

print("BM25 #1:", corpus[bm[0][0]][:50])
print("Dense #1:", corpus[dn[0][0]][:50])
print("Fused #1:", corpus[fused[0][0]][:50])
```

Find a second query where BM25 and dense disagree more sharply — where the fused winner is not the winner of either individual retriever. Report that query and the rank positions.

**Medium.** Replace RRF with a weighted linear combination of min-max normalized scores. Implement the alpha sweep and measure recall@5 across a held-out query set with relevance labels you define. The following code is complete and runnable:

```python
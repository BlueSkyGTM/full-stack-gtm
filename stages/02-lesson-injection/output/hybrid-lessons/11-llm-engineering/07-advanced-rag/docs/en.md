# Advanced RAG (Chunking, Reranking, Hybrid Search)

## Learning Objectives

- Implement three chunking strategies (fixed-size, recursive, sentence-based) and diagnose where each severs critical information at chunk boundaries
- Build a hybrid search pipeline combining BM25 sparse retrieval with dense vector retrieval, fused via Reciprocal Rank Fusion (RRF)
- Apply a cross-encoder reranker to a candidate set and measure the ranking delta between bi-encoder retrieval and cross-encoder scoring
- Diagnose RAG failure modes where naive top-k retrieval returns semantically similar but factually irrelevant chunks

## The Problem

You built a basic RAG pipeline. It embeds documents, stores them in a vector database, and retrieves the top-k most similar chunks for a given query. On a small corpus with straightforward questions, it works. Scale it up and three failure patterns emerge repeatedly.

The first failure is chunk boundaries splitting facts. Your embedding model chunks a 10-K filing into 512-token windows. The sentence "Acme Corp reported $47.2M in Q3 2025 earnings, a 23% increase driven by EMEA expansion" gets cut between "$47.2M" and "in Q3 2025." Now neither chunk contains the full fact. The retrieval system returns a chunk with a dollar amount but no quarter, and another chunk with a quarter but no dollar amount. The LLM cannot synthesize the answer because the full fact never existed in a single retrieved chunk.

The second failure is exact-match blindness. Dense vector retrieval captures semantic similarity — it knows that "revenue" and "earnings" are related concepts. But it cannot reliably distinguish between part numbers, company names, or specific identifiers. A query for "AC-7742-B" returns chunks about products in general, not the chunk containing that exact part number. The embedding space smooths over lexical specificity that matters enormously in practice.

The third failure is ranking quality. Your vector store returns the top-5 most similar chunks by cosine similarity. But "most similar" by embedding distance is not "most useful for answering this question." A chunk that shares vocabulary with the query but answers a different question ranks higher than the chunk that directly answers it. The embedding model encodes semantic relatedness, not answer relevance — and those are different things.

## The Concept

Three mechanisms address these failure modes. Each targets a specific bottleneck in the retrieve-then-generate pipeline.

**Chunking strategy** determines where document boundaries fall. Fixed-size chunking (every N characters or tokens) is the simplest approach but makes no attempt to respect sentence, paragraph, or semantic boundaries. Recursive character splitting tries progressively smaller separators — paragraph breaks, then sentence boundaries, then word boundaries — to keep related text together while staying within a size budget. Sentence-based chunking treats each sentence as an atomic unit, which preserves facts but can produce chunks too small for useful context. The chunk boundary problem is the root cause behind most "RAG doesn't work" complaints: the information exists in your corpus but no single chunk contains enough of it to be retrieved and used.

**Hybrid search** combines two retrieval paradigms that fail in complementary ways. Dense vector retrieval (your embedding model) captures semantic similarity but misses exact keyword matches. Sparse retrieval via BM25 captures lexical matches — term frequency, inverse document frequency — but misses synonyms and paraphrases. The query "companies expanding into Europe" needs semantic matching to find "EMEA growth strategy," but the query "NAICS code 541512" needs exact lexical matching. Reciprocal Rank Fusion (RRF) merges both result sets without requiring score normalization: for each document, sum `1 / (k + rank)` across both ranked lists, where `k` is typically 60. Documents that appear highly in both lists get boosted; documents that appear highly in only one list get a moderate score. The constant `k` dampens the advantage of being rank 1 versus rank 2, preventing one retrieval method from dominating.

**Two-stage reranking** separates the retrieval problem from the ranking problem. A bi-encoder (your vector store) embeds the query and each document independently, then compares via cosine similarity. This is fast — O(n) comparisons after pre-computing document embeddings — but the query and document never interact during encoding. A cross-encoder encodes the query and document together, with full attention between all tokens in both texts. The model sees "What was Q3 revenue?" and "$47.2M in Q3 2025 earnings" simultaneously and can directly assess whether the document answers the question. This is precise but O(n) forward passes for n candidates, making it too slow for full-corpus retrieval. The retrieve-then-rerank pattern uses the fast bi-encoder to fetch 50-100 candidates, then the slow cross-encoder to re-score and re-rank them.

```mermaid
flowchart TD
    A[Document Corpus] --> B[Chunking Strategy<br/>fixed / recursive / semantic]
    B --> C1[BM25 Sparse Index]
    B --> C2[Dense Vector Index]
    Q[User Query] --> D1
    Q --> D2
    C1 --> D1[BM25 Retrieval<br/>top-k candidates]
    C2 --> D2[Dense Retrieval<br/>top-k candidates]
    D1 --> E[RRF Fusion<br/>score = Σ 1/(k + rank)]
    D2 --> E
    E --> F[Top-N Candidates<br/>N=20 to 100]
    F --> G[Cross-Encoder Reranker<br/>query ↔ doc full attention]
    G --> H[Final Top-K Results<br/>k=3 to 5]
```

## Build It

Build a minimal pipeline that demonstrates all three mechanisms on the same document corpus. Each section produces observable output so you can see where chunking severs facts, where hybrid search outperforms either method alone, and how much the cross-encoder reorders results.

### Part 1: Chunking Comparison

```python
import re

document = (
    "Acme Corp reported Q3 2025 earnings of $47.2M, a 23% increase. "
    "The EMEA segment drove growth with European revenue reaching $18.6M. "
    "CEO Jane Doe attributed the expansion to acquisitions in Germany and France. "
    "The board approved a $50M share buyback program. "
    "CFO John Smith noted operating margins improved to 34% from 29%. "
    "Acme competes with Globex Corp which reported Q3 revenue of $41.1M. "
    "Part number AC-7742-B represents the flagship enterprise product line contributing $12.3M quarterly. "
    "The company is headquartered in Austin Texas with 2,400 employees."
)

def fixed_size_chunks(text, size=150):
    return [text[i:i+size] for i in range(0, len(text), size)]

def recursive_chunks(text, max_size=150):
    sentences = text.split('. ')
    chunks = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = (current + ". " + sent) if current else sent
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(sent) <= max_size:
                current = sent
            else:
                words = sent.split(' ')
                current = ""
                for word in words:
                    test = (current + " " + word) if current else word
                    if len(test) <= max_size:
                        current = test
                    else:
                        if current:
                            chunks.append(current)
                        current = word
    if current:
        chunks.append(current)
    return chunks

def sentence_chunks(text, max_sentences=2):
    sentences = [s.strip() + '.' for s in text.split('. ') if s.strip()]
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(' '.join(sentences[i:i+max_sentences]))
    return chunks

print("=== FIXED-SIZE (150 chars) ===")
for i, c in enumerate(fixed_size_chunks(document)):
    print(f"  [{i}] ...{c[-40:]}")
    if '$' in c and 'Q3' not in c and 'earnings' in c:
        print(f"      ⚠ FACT SEVERED: dollar amount split from quarter context")

print("\n=== RECURSIVE (150 chars) ===")
for i, c in enumerate(recursive_chunks(document)):
    print(f"  [{i}] ({len(c)} chars) {c[:60]}...")

print("\n=== SENTENCE-BASED (2 sentences) ===")
for i, c in enumerate(sentence_chunks(document)):
    print(f"  [{i}] {c[:60]}...")
```

Run it and observe: fixed-size splitting severs the Q3 earnings fact. Recursive splitting keeps sentences intact. Sentence-based chunking preserves every fact but produces more, smaller chunks — increasing retrieval cost and surface area.

### Part 2: Hybrid Search with RRF

```python
import math
from collections import Counter

chunks = recursive_chunks(document, max_size=120)

def bm25_score(query, docs, k1=1.5, b=0.75):
    tokenized = [d.lower().split() for d in docs]
    avgdl = sum(len(d) for d in tokenized) / len(tokenized)
    df = Counter()
    for doc in tokenized:
        for term in set(doc):
            df[term] += 1
    scores = []
    q_terms = query.lower().split()
    for doc in tokenized:
        tf = Counter(doc)
        score = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            idf = math.log((len(docs) - df[term] + 0.5) / (df[term] + 0.5) + 1)
            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * len(doc) / avgdl)
            score += idf * numerator / denominator
        scores.append(score)
    return scores

def dense_sim(query, docs):
    def vec(text):
        tokens = text.lower().split()
        return Counter(tokens)
    qv = vec(query)
    results = []
    for d in docs:
        dv = vec(d)
        dot = sum(qv[t] * dv[t] for t in qv)
        mag = math.sqrt(sum(v**2 for v in qv.values())) * math.sqrt(sum(v**2 for v in dv.values()))
        results.append(dot / mag if mag > 0 else 0.0)
    return results

def rrf_fuse(bm25_ranks, dense_ranks, k=60):
    n = len(bm25_ranks)
    scores = [0.0] * n
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] += 1.0 / (k + rank + 1)
    for rank, idx in enumerate(dense_ranks):
        scores[idx] += 1.0 / (k + rank + 1)
    return scores

query_exact = "AC-7742-B"
query_semantic = "European expansion strategy"

for q in [query_exact, query_semantic]:
    bm25 = bm25_score(q, chunks)
    dense = dense_sim(q, chunks)
    bm25_ranked = sorted(range(len(chunks)), key=lambda i: bm25[i], reverse=True)
    dense_ranked = sorted(range(len(chunks)), key=lambda i: dense[i], reverse=True)
    rrf = rrf_fuse(bm25_ranked, dense_ranked)
    rrf_ranked = sorted(range(len(chunks)), key=lambda i: rrf[i], reverse=True)

    print(f"\n=== QUERY: '{q}' ===")
    print(f"  BM25 #1:    chunk[{bm25_ranked[0]}] = {chunks[bm25_ranked[0]][:50]}...")
    print(f"  Dense #1:   chunk[{dense_ranked[0]}] = {chunks[dense_ranked[0]][:50]}...")
    print(f"  RRF #1:     chunk[{rrf_ranked[0]}] = {chunks[rrf_ranked[0]][:50]}...")
```

The exact-match query for "AC-7742-B" should rank the part-number chunk first under BM25. Dense similarity alone may bury it because the token frequency of "AC-7742-B" is meaningless in a cosine space dominated by common words. RRF combines both signals.

### Part 3: Cross-Encoder Reranking

```python
def cross_encoder_score(query, doc):
    q_tokens = set(query.lower().split())
    d_tokens = set(doc.lower().split())
    overlap = q_tokens & d_tokens
    coverage = len(overlap) / len(q_tokens) if q_tokens else 0
    position_bonus = 0.0
    doc_lower = doc.lower()
    for qt in q_tokens:
        idx = doc_lower.find(qt)
        if idx >= 0:
            position_bonus += 1.0 / (1 + idx / 100)
    return coverage * 0.6 + (position_bonus / max(len(q_tokens), 1)) * 0.4

query = "What were Q3 2025 earnings?"
dense = dense_sim(query, chunks)
dense_ranked = sorted(range(len(chunks)), key=lambda i: dense[i], reverse=True)[:5]

print(f"\n=== RERANKING for '{query}' ===")
print(f"{'Rank':<6}{'Bi-encoder':<40}{'Cross-encoder':<40}")
print("-" * 86)
reranked = sorted(dense_ranked, key=lambda i: cross_encoder_score(query, chunks[i]), reverse=True)
for rank in range(5):
    bi_idx = dense_ranked[rank]
    ce_idx = reranked[rank]
    bi_preview = chunks[bi_idx][:35].replace('\n', ' ')
    ce_preview = chunks[ce_idx][:35].replace('\n', ' ')
    moved = "" if bi_idx == ce_idx else f"  (was #{dense_ranked.index(ce_idx)+1})"
    print(f"{rank+1:<6}{f'[{bi_idx}] {bi_preview}...':<40}{f'[{ce_idx}] {ce_preview}...{moved}':<40}")
```

The cross-encoder reorders candidates. A chunk that shares vocabulary but does not contain the earnings figure drops. The chunk with "$47.2M" and "Q3 2025" in close proximity rises.

## Use It

This cross-encoder reranking pipeline (BM25 sparse + dense cosine fused via RRF, then re-scored by pairwise query-document attention) maps directly to account research enrichment workflows where you query a corpus of 10-K filings, earnings transcripts, and press releases for specific account intelligence.

```python
account_corpus = [
    "Globex Corp Q3 2025 revenue $41.1M down 4% YoY due to APAC contraction",
    "Globex announced acquisition of DataFlow Systems for $89M cash October 2025",
    "Globex CEO Mark Chen cited FX headwinds and deferred enterprise renewals",
    "Initech LLC reported Q3 revenue $22.0M with 12% growth in mid-market segment",
    "Globex Corp board authorized $30M buyback citing undervalued share price",
]
query = "Globex Corp revenue decline"

bm25 = bm25_score(query, account_corpus)
dense = dense_sim(query, account_corpus)
bm25_r = sorted(range(len(account_corpus)), key=lambda i: bm25[i], reverse=True)
dense_r = sorted(range(len(account_corpus)), key=lambda i: dense[i], reverse=True)
rrf = rrf_fuse(bm25_r, dense_r)
candidates = sorted(range(len(account_corpus)), key=lambda i: rrf[i], reverse=True)[:3]
final = sorted(candidates, key=lambda i: cross_encoder_score(query, account_corpus[i]), reverse=True)

print(f"Query: '{query}'\n")
for rank, idx in enumerate(final):
    print(f"  #{rank+1} (score={cross_encoder_score(query, account_corpus[idx]):.3f}) {account_corpus[idx]}")
```

For a GTM team building account intelligence enrichment, this pipeline retrieves the right context from filings for a specific named account [CITATION NEEDED — concept: GTM cluster mapping for account research RAG]. The hybrid search catches the exact company name ("Globex Corp") via BM25, and the cross-encoder prioritizes the chunk that actually discusses the revenue decline over the one that merely mentions the company name in a different context (the buyback announcement). In production, swap the toy `cross_encoder_score` for `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers and the BM25 implementation for `rank_bm25` or Elasticsearch.

## Exercises

**Exercise 1 (Easy):** Modify the fixed-size chunker to use a 300-character window instead of 150. Re-run Part 1. Does the Q3 earnings fact still get severed? At what window size does it stop? Document the threshold and explain why recursive chunking avoids the problem regardless of window size.

**Exercise 2 (Hard):** Replace the toy `cross_encoder_score` function with a real cross-encoder from the `sentence-transformers` library (`CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`). Run the Part 3 reranking on the same corpus and queries. Measure the Kendall rank correlation between the toy scorer's ranking and the real model's ranking. Where do they disagree, and which ranking is more correct for answer relevance? Write a one-paragraph diagnosis of what the real cross-encoder captures that the toy scorer cannot.

## Key Terms

- **BM25 (Best Match 25):** A sparse retrieval scoring function that ranks documents by term frequency, inverse document frequency, and document length normalization. Excels at exact keyword matching.
- **Reciprocal Rank Fusion (RRF):** A score fusion method that combines multiple ranked lists by summing `1/(k + rank)` for each document across lists. Requires no score normalization between retrieval methods.
- **Cross-encoder:** A model that encodes a query and document together in a single forward pass with full cross-attention. Slower than bi-encoder retrieval but produces more accurate relevance scores.
- **Bi-encoder:** A model that encodes query and document independently into vectors, then compares via cosine similarity. Enables fast retrieval via approximate nearest neighbor search but misses query-document interaction.
- **Recursive character splitting:** A chunking strategy that attempts splits at progressively smaller boundaries (paragraph → sentence → word) to keep related text in the same chunk while respecting a size limit.
- **Chunk boundary severance:** The failure mode where a fact spanning two chunks becomes unretrievable because neither chunk contains enough context to be matched and used independently.

## Sources

- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.* SIGIR 2009.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. (Bi-encoder architecture for efficient retrieval)
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085. (Cross-encoder application for document reranking)
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. (Retrieve-then-rerank pattern with dense and sparse retrieval)
- [CITATION NEEDED — concept: GTM cluster mapping for account research RAG applications]
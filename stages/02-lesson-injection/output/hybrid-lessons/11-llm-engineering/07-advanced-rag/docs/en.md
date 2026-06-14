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
                    test =
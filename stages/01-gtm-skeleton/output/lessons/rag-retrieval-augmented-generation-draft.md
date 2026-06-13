# RAG (Retrieval-Augmented Generation)

## Beat 1: Hook — "The LLM Doesn't Know Your Data"

LLMs freeze at training time. They cannot cite your playbook, your accounts, or your internal research. RAG is the engineering pattern that bridges that gap: retrieve relevant context from your own corpus, inject it into the prompt, and let the model generate grounded on evidence rather than hallucination.

## Beat 2: Concept — "Retrieve, Then Generate"

The RAG pattern has three stages: (1) embed a query into the same vector space as your document chunks, (2) retrieve the top-k most similar chunks via cosine similarity, and (3) concatenate those chunks into the prompt as context before calling the generator. No fine-tuning required — the model's weights stay frozen; only the prompt changes per query.

## Beat 3: Mechanism — "From Text to Vectors to Answers"

Cover the full pipeline: document ingestion → chunking (fixed-size vs. semantic) → embedding each chunk → storing in a vector index → at query time, embedding the query → approximate nearest neighbor search → constructing the augmented prompt → calling the LLM. Explain chunk overlap, why cosine similarity works for normalized embeddings, and where retrieval fails (mismatched granularity, stale index, embedder-drift). Tool mentions: `text-embedding-3-small` for embeddings, `numpy` dot-product for search, `Chroma` as a persistent vector store.

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def embed(texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in resp.data])

docs = [
    "Clay waterfall enriches leads by sequencing data providers.",
    "RAG retrieves relevant context before generating a response.",
    "Vector databases index embeddings for fast similarity search.",
    "Chunking splits documents into retrievable segments.",
    "Cosine similarity measures alignment between two vectors."
]

doc_embeddings = embed(docs)
query_embedding = embed(["How does data enrichment work?"])

similarities = doc_embeddings @ query_embedding.T
ranked = np.argsort(similarities.flatten())[::-1]

for i in ranked[:3]:
    print(f"Score: {similarities.flatten()[i]:.4f} | {docs[i]}")
```

## Beat 4: Use It — "RAG in GTM: Research Agents and Knowledge Retrieval"

GTM cluster: **Zone 03 (Research)** and **Zone 02 (Enrichment)**. RAG is the retrieval backbone of research agents that query your internal playbooks, past deal notes, and competitive intel before generating account briefs or outreach. The Clay waterfall enrichment pattern is complementary — waterfall fetches structured data from external providers; RAG fetches unstructured context from your own corpus. Both feed the same generation step.

**Exercise hooks:**
- Easy: Embed a 10-sentence sales playbook and retrieve the top-3 chunks for a given account question.
- Medium: Build a minimal RAG CLI that accepts a query, retrieves from a pre-built index of deal notes, and prints the generated answer with source citations.
- Hard: Implement hybrid retrieval (dense embedding + keyword BM25), compare results against dense-only on 5 test queries, and print precision@3 for each strategy.

## Beat 5: Ship It — "Build a Working RAG Pipeline Over a Real Corpus"

Write a complete, runnable script that: ingests a folder of markdown files, chunks them with 200-token overlap, embeds them, stores in a numpy array (or Chroma if available), accepts a user query from stdin, retrieves top-k, constructs a prompt with retrieved context, calls the LLM, and prints the answer with chunk sources. No scaffolding — the code runs end-to-end.

## Beat 6: Evaluate — "Where RAG Breaks"

Quiz questions target the mechanism, not trivia. Sample angles: why does cosine similarity fail when query and document use different terminology for the same concept? What happens to retrieval quality when chunks are too large? If the embedder used at query time differs from the one used at index time, what breaks and why? Faithfulness vs. relevance as dual evaluation axes.

---

**Learning Objectives:**

1. Build a retrieval pipeline that embeds documents, indexes them, and returns the top-k most similar chunks for a given query.
2. Construct a RAG prompt that concatenates retrieved context with a user question and passes it to a generator.
3. Compare chunking strategies (fixed-size, sentence-boundary, semantic) and their impact on retrieval precision.
4. Diagnose retrieval failures: embedder mismatch, stale index, granularity misalignment, vocabulary gap.
5. Evaluate RAG output on two axes: faithfulness (is the answer grounded in retrieved context?) and relevance (did retrieval return the right context?).
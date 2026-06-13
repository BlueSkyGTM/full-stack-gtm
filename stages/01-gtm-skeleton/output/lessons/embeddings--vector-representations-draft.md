# Embeddings & Vector Representations

## Hook

You've been matching strings with regex and fuzzy logic. That breaks the moment a prospect says "revenue ops" instead of "RevOps." Embeddings convert text into coordinate positions in mathematical space — where synonyms land next to each other and "we need help with pipeline" finally matches "CRM optimization request."

## Concept

A word or sentence becomes a fixed-length array of floats. The geometric distance between two vectors encodes semantic similarity: cosine close to 1 means same meaning, close to 0 means unrelated, close to -1 means opposed. The embedding model is a trained neural net that maps tokens to that space. Everything downstream — similarity search, clustering, classification — is just linear algebra on those arrays.

## Build

Generate embeddings from raw text using a local model (`sentence-transformers`), compute pairwise cosine similarity, and print the scores so you can see "sales pipeline" score higher against "revenue funnel" than against "data engineering."

- **Easy:** Embed three sentences and print which pair is closest.
- **Medium:** Build a function that takes a query and ranks a list of 20 phrases by cosine similarity, printing the top 5 with scores.
- **Hard:** Implement brute-force nearest-neighbor search over 100+ embedded records, return top-k results, and benchmark against a naive loop to establish your baseline latency number.

## Use It

Every inbound lead comes with free-text: form comments, email bodies, LinkedIn messages. Keyword routing misses the edge cases. Embed each incoming message, compare it against pre-embedded sequence category vectors (e.g., "sales engagement," "technical evaluation," "budget discussion"), and route to the sequence with highest cosine similarity. This is the Signal Machine inside Inbound-Led Outbound (3.3) — semantic routing before the lead goes cold.

- **Easy:** Embed five example inbound messages and three sequence category labels; print the similarity matrix.
- **Medium:** Write a router function that accepts any inbound string, embeds it, and returns the best-matching sequence category with confidence score.
- **Hard:** Feed 50 historical inbound messages through the router, compare automated assignments against human-labeled ground truth, and print accuracy, precision, and recall per category.

## Ship It

Production embedding pipelines need to handle scale, cost, and model consistency. Cover: batch embedding to amortize API overhead, caching vectors so you don't re-embed unchanged records, and pinning a model version so your similarity thresholds don't drift when the provider updates weights.

- **Easy:** Write a script that embeds 50 texts in one batch call and prints per-item latency plus total time.
- **Medium:** Implement a file-based embedding cache (hash input text → stored vector) and demonstrate cache-hit speedup on a second run.
- **Hard:** Build a versioned embedding store that detects when the model identifier changes and forces a full re-embed, logging the migration event with timestamp and record count.

## Extend

Vector databases (Pinecone, Weaviate, Qdrant, Chroma) index embeddings for approximate nearest-neighbor search at million-record scale. Dense + sparse hybrid retrieval (hybrid search) combines keyword precision with semantic recall. Retrieval-Augmented Generation chains embedding search with LLM generation — but that's a later lesson, and the embedding foundation has to hold first.

- **Easy:** Install Chroma, store 10 embedded documents, and query for the top 3 nearest neighbors, printing results.
- **Medium:** Implement the same top-k search using Pinecone's free tier and compare returned rankings against your local brute-force results.
- **Hard:** Build a hybrid search function that combines BM25 keyword scores with cosine similarity scores using a weighted merge, then evaluate whether recall improves over pure semantic search on a labeled test set.

---

**Learning Objectives (for `docs/en.md`):**

1. Generate embeddings from raw text using a sentence-transformer model and inspect the resulting vector dimensions.
2. Compute cosine similarity between two embedding vectors and interpret the score as semantic relatedness.
3. Implement a semantic router that maps inbound free-text to the closest predefined category by vector distance.
4. Evaluate routing accuracy against human-labeled ground truth using precision, recall, and accuracy metrics.
5. Configure a batch embedding pipeline with caching and model-version pinning for production use.

**GTM Redirect:** Inbound-Led Outbound (3.3) — Signal Machine. The embedding model routes inbound leads to the correct outreach sequence based on semantic meaning, not keyword matching. This is the mechanism that makes "we need to fix our funnel" land in the same bucket as "pipeline optimization" without maintaining an exhaustive synonym list.
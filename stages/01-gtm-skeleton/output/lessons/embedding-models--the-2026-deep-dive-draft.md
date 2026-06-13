# Embedding Models — The 2026 Deep Dive

## Hook It

A practitioner has seen vector search in product demos or API docs. This beat anchors on a concrete frustration: keyword search fails on "pricing" when the page says "investment required." Embeddings solve this by mapping semantically similar text to nearby points in vector space — even when zero tokens overlap.

## Ground It

Mechanism first: an embedding model is a trained neural network that compresses text into a fixed-length float array where cosine similarity approximates human-judged semantic relatedness. Cover the three lineage families dominant in 2026 — BGE/MISTRAL-based, proprietary (Voyage, Cohere), and open-weight (Nomic, GTE-Qwen2) — and explain *why* they produce different results: contrastive training objectives, negative-sample mining strategy, and matryoshka dimensionality. Explain cosine similarity as a dot product of normalized vectors, not "magic." Address the skeptics: embeddings are lossy, deterministic, and position-sensitive — show where they break (negation, temporal order, long-tail jargon).

## Build It

Runnable Python script (no edits needed) that: (1) calls a local sentence-transformers model to embed a corpus of 12 GTM-flavored sentences, (2) computes pairwise cosine similarity with pure NumPy, (3) prints a ranked retrieval result for a query, and (4) demonstrates the "pricing" vs "investment required" overlap example from the hook. All output is terminal-visible. Exercise hooks: Easy — swap the query string and re-run; Medium — add 10 more sentences and observe ranking drift; Hard — implement brute-force k-NN and time it against a toy FAISS index for the same corpus.

## Use It

GTM redirect → **Cluster 06: Inbound-Led Outbound (3.3) / Signal Machine.** Demonstrate the mechanism: embed every inbound lead's submitted text (form fills, email bodies, chat transcripts) and compare against pre-labeled sequence embeddings. The practitioner sees how cosine score thresholds route leads to "demo request," "technical question," or "partner inquiry" sequences without regex or keyword lists. Exercise hooks: Easy — classify 5 lead texts into 3 pre-defined buckets by similarity; Medium — tune the cosine threshold and measure precision/recall on a labeled dataset of 40 leads; Hard — build a two-stage retriever: embeddings for recall, then an LLM judge for precision.

## Ship It

Production concerns, not theory. Cover: embedding model versioning (regenerate means all vectors change), batching strategies for rate limits, dimension reduction (matryoshka) for storage savings, and the hard truth that embeddings are not a database — you still need metadata filtering alongside vector search. Provide a runnable script that writes embeddings + metadata to a local SQLite + `sqlite-vss` table and queries it. Exercise hooks: Easy — insert 50 records and run 3 queries; Medium — add a metadata pre-filter before vector search and measure latency difference; Hard — implement a crude drift detector that flags when new inbound text has no neighbor above 0.6 cosine similarity.

## Prove It

Five quiz questions grounded in the mechanisms above: (1) calculate cosine similarity between two provided 4-d vectors by hand, (2) identify which training objective produces better negative-sample discrimination, (3) predict ranking change when matryoshka truncation drops from 1024 to 256 dimensions, (4) diagnose why a "partner inquiry" lead routed to "demo request" — is it threshold, corpus, or model version, (5) explain why regenerating embeddings for one document requires regenerating all. All questions reference code or concepts from Ground It / Build It / Use It.
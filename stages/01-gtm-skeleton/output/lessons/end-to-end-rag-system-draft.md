# End-to-End RAG System

## Hook It

You have documents. You have questions. The LLM has neither. RAG bridges that gap — retrieve the right context, inject it into the prompt, and ground the generation in facts you control rather than weights you don't.

## Ground It

Prerequisites this lesson assumes: embedding vectors, cosine similarity, tokenization, basic prompt construction. If any of those are shaky, revisit earlier modules before proceeding.

## Explain It

The RAG pipeline is five stages: (1) **Ingest** — load raw documents from files, APIs, or databases. (2) **Chunk** — split documents into passages small enough to embed but large enough to carry meaning. (3) **Index** — embed each chunk and store it in a vector store with metadata. (4) **Retrieve** — embed the user query, run similarity search against the index, collect top-k candidates. (5) **Generate** — stuff retrieved chunks into the LLM prompt alongside the question, ask for a grounded answer. Each stage has failure modes: bad chunk boundaries lose context, missing metadata prevents filtering, low-k retrieval misses the right passage, over-stuffed context dilutes the signal. The mechanism is deterministic retrieval feeding probabilistic generation.

## Show It

Build a complete RAG pipeline in a single Python script: load `.txt` files from a directory, chunk by character count with overlap, embed using `sentence-transformers`, store in a `chromadb` collection, accept a query, retrieve top-3 chunks, construct a prompt, call the LLM, print both retrieved context and final answer. All code runs in terminal. Observable output: the retrieved chunk texts print before the final answer so you can verify retrieval quality independently from generation quality.

## Use It

In GTM, RAG powers internal knowledge bases — sales enablement bots that answer "what's our pricing for enterprise tier" from your actual pricing docs, not from the LLM's training data. Maps to **Zone 30 (Closing/Expansion)** and **Zone 20 (Pipeline)** as a sales-engineering enablement tool. [CITATION NEEDED — concept: RAG-based GTM knowledge base integration with Clay workflows]. If you're building prospect research agents, the same retrieval mechanism surfaces relevant case studies or competitor intel per-account.

## Ship It

- **Easy**: Run the provided pipeline on a set of 5 sample documents. Change the chunk size from 500 to 200 characters. Print the retrieved chunks and observe how retrieval quality shifts.
- **Medium**: Add metadata filtering — tag each chunk with a `source` field, then restrict retrieval to a single source at query time. Print the filter condition and the retrieved chunks to confirm filtering works.
- **Hard**: Implement a reranker. After initial top-10 retrieval, score each chunk with a cross-encoder (`sentence-transformers` provides one), re-rank by cross-encoder score, and pass only top-3 to the LLM. Print pre-rerank order vs. post-rerank order to observe the reordering.
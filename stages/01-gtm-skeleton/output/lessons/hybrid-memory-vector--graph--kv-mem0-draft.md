# Hybrid Memory: Vector + Graph + KV (Mem0)

---

## Beat 1: Hook

You've tried pure vector retrieval. Sometimes the top-k results miss the *relationship* between facts — "Alice works at Acme" and "Acme uses Salesforce" sit at unrelated positions in embedding space even though they're one hop apart. Pure graph or pure KV has the inverse problem: exact matches only, no fuzzy recall. This lesson builds a three-layer memory system that queries all three and merges results.

---

## Beat 2: Concept

Describe the three memory primitives — vector (similarity), graph (traversal), KV (exact lookup) — and where each fails alone. Introduce the hybrid merge pattern: parallel queries to each store, followed by a deduplication and ranking step. Then name the tool: Mem0 implements this pattern as an API with automatic extraction of entities and facts from conversation text.

---

## Beat 3: Mechanism

Walk through Mem0's pipeline: (1) text in → (2) LLM extracts entities + relations → (3) entities written to graph, raw text embedded to vector store, structured attributes to KV → (4) query fan-out to all three → (5) score fusion and return. Show the flow as a diagram-style sequence. Then demonstrate with a runnable Python script that adds memories and retrieves them, printing the source store for each result so the practitioner can observe which layer answered which query.

---

## Beat 4: Use It

**GTM Redirect:** This is the memory architecture behind multi-touch sequence personalization in Zone 2 enrichment workflows. When your Clay waterfall enriches a prospect across multiple runs, hybrid memory is what lets you recall "prospect mentioned Q4 budget review" (vector), "prospect reports to VP Engineering" (graph), and "company = Acme Corp" (KV) from a single query.

Exercise hooks:
- **Easy:** Add 5 conversational facts about a prospect via the Mem0 API, then query for "budget timeline" and print which memory layer returned the hit.
- **Medium:** Ingest a mock sales call transcript, extract entities, run a graph traversal query to find the decision-maker chain, and print the path.
- **Hard:** Build a minimal score-fusion function that takes ranked lists from vector, graph, and KV, merges by reciprocal rank, and compare its top-3 recall against vector-only retrieval on the same dataset.

---

## Beat 5: Ship It

Describe the production deployment shape: Mem0 as a Docker container or managed API, a backing Postgres + Neo4j (or Mem0's bundled stores), and an ingestion hook wired into whatever agent or workflow emits text. Cover the operational concerns — extraction cost per message (LLM call for entity extraction), latency of the fan-out query, and cache strategy for hot entities. Provide a runnable script that spins up the client against the hosted API, ingests a batch, and benchmarks retrieval latency.

Exercise hooks:
- **Easy:** Configure a Mem0 client with your API key, add 10 memories in batch, and print the add-then-query round-trip latency.
- **Medium:** Write a script that ingests 50 messages, then runs 10 retrieval queries, logging per-query latency and which store returned the top result.
- **Hard:** Implement a lightweight cache layer (Python dict with TTL) in front of Mem0's retrieval, measure the latency reduction on repeated queries, and print the hit-rate ratio.

---

## Beat 6: Evaluate

Three recall-quality checks for hybrid memory: (1) precision@k — do returned memories actually answer the query? (2) coverage — does the graph layer capture the entity relations you'd expect from the source text? (3) redundancy — are KV and graph returning overlapping facts that should be deduplicated? Provide a scoring script that runs a labeled test set and prints all three metrics.

Exercise hooks:
- **Easy:** Hand-label 5 query–expected-memory pairs, run retrieval, and print precision@3.
- **Medium:** Generate a synthetic dataset of 20 entity-relation facts, ingest them, query for each relation, and compute graph coverage (fraction of relations successfully retrieved).
- **Hard:** Run the full evaluation suite — precision@k, graph coverage, and redundancy rate — against two configurations (vector-only vs. hybrid) and print a side-by-side comparison table.

---

## Learning Objectives (draft)

1. **Compare** retrieval failure modes across vector-only, graph-only, and KV-only memory stores.
2. **Implement** a three-store hybrid retrieval query with score fusion in Python.
3. **Configure** Mem0's client to extract entities from conversational text and persist to all three layers.
4. **Evaluate** recall quality of hybrid memory using precision@k, coverage, and redundancy metrics.
5. **Diagnose** which memory layer is answering which query type by tracing result provenance.
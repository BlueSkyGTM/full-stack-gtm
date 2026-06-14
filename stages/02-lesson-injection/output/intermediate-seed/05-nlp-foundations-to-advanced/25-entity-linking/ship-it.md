## Ship It

Production entity linking has four failure modes that determine whether your pipeline survives contact with real data.

**Latency budget.** Candidate generation is the bottleneck. If your alias index has 10 million entries and you are linking 500 mentions per document, querying the index 500 times adds up. The standard mitigation is an in-memory alias dictionary (for surface forms that fit in RAM) combined with an approximate nearest neighbor index (FAISS, ScaNN) for embedding-based candidate retrieval. TF-IDF linkers are fast — sub-millisecond per mention — because they compute cosine similarity on sparse vectors. Embedding-based linkers add 20–50ms per mention for the encoder forward pass. Generative linkers (GENRE, LLM-based) are 200ms+ per mention, which rules them out for high-throughput enrichment pipelines.

**Caching surface forms.** Many mentions repeat across documents: "Apple," "Google," "Microsoft" appear constantly. Cache the linked result keyed by `(surface_form, context_hash)` or just `surface_form` for high-prior entities. A simple LRU cache on surface forms typically handles 60–80% of lookups in production GTM data, because company name distributions are heavy-tailed — the top 1000 company names account for the majority of mentions.

**NIL handling.** When no candidate exceeds the threshold, the linker emits NIL (no match). NIL entities are not errors — they are signal. A NIL on "DataLayer AI" means the company is not in your KB yet. In a Clay workflow, NIL should trigger a separate path: a manual review queue, a Google search enrichment, or a new-entity creation flow. The worst production failure is forcing a link below threshold — you get a confident wrong answer instead of an honest "I don't know." Set the threshold using evaluation data (below), not intuition.

**Stale knowledge bases.** Companies merge, rename, and dissolve. A KB built in January will miss companies founded in March and will still link "Twitter" to the pre-rebrand entity instead of "X." Schedule periodic KB rebuilds — monthly at minimum for GTM use cases, where new companies appear weekly. The rebuild is cheap if your KB sources are API-accessible (Companies House, SEC filings, Crunchbase). The expensive part is maintaining the alias dictionary: you need someone or some process to add "X" as an alias for the entity formerly known as "Twitter."

**Evaluation.** Build a held-out set of 200–500 annotated mentions with gold-standard entity IDs. Compute micro-precision and micro-recall:

- **Micro-precision:** of all linked entities, what fraction match the gold standard?
- **Micro-recall:** of all gold-standard entities, what fraction did the linker correctly link?

Micro-precision catches false links (linking to the wrong entity). Micro-recall catches missed links (emitting NIL when a valid entity exists). Track both across KB rebuilds and linker changes. A new alias that improves recall by 5% but drops precision by 2% may or may not be worth it depending on your downstream cost of wrong-row enrichment versus missed enrichment.
# Chunking Strategies, Compared

---

## Learning Objectives

1. Compare fixed-size, recursive, sentence-based, and semantic chunking by their boundary behavior and retrieval implications
2. Implement each chunking strategy in Python and print chunk count, size distribution, and boundary content
3. Evaluate which strategy fits a given document type based on its structure
4. Configure overlap parameters and predict their effect on recall
5. Detect when semantic chunking adds cost without retrieval improvement

---

## Beat 1: Hook

The chunk boundary *is* the retrieval boundary. If your chunks split mid-idea, your retriever will surface half an answer regardless of embedding quality. The choice of chunking strategy determines what the retriever can and cannot find.

---

## Beat 2: Concept (Mechanism First)

Five strategies, each solving a different boundary problem:

- **Fixed-size**: splits every N characters. Predictable size, no semantic awareness. Breaks words and sentences at boundaries.
- **Recursive character**: splits by a hierarchy of separators (`\n\n` → `\n` → `. ` → ` `) with configurable overlap. Attempts to keep paragraphs, then sentences, then words intact before falling back.
- **Sentence-based**: uses an NLP sentence tokenizer to split at sentence boundaries. Preserves grammatical units but produces variable-length chunks.
- **Semantic**: embeds each sentence, computes cosine similarity between adjacent sentences, and splits where similarity drops below a threshold. Expensive — one embedding call per sentence.
- **Document-structure**: splits on markup (Markdown headers, HTML tags, JSON keys). Best for structured sources where headings indicate topic boundaries.

Tradeoff axes: computational cost, chunk size variance, semantic coherence at boundaries, sensitivity to document structure.

---

## Beat 3: Code (Working Examples)

Four runnable scripts. Each takes the same source text, applies one strategy, and prints:
- Total chunk count
- Min/mean/max chunk length
- First 80 characters of chunks 1, 3, and last (to inspect boundaries)

Strategies implemented:
1. Fixed-size (raw Python slicing)
2. Recursive character with overlap (LangChain's `RecursiveCharacterTextSplitter`)
3. Sentence-based (spaCy or NLTK sentence tokenizer)
4. Semantic (embed-adjacent-compare-split loop using `sentence-transformers`)

Each script runs standalone. Each prints observable output.

---

## Beat 4: Use It (GTM Redirect)

**GTM Cluster: Enablement Knowledge Base**

When a rep asks "what's our competitive positioning against [competitor]?" the retriever pulls chunks from your competitive intel docs. If those chunks were split at fixed 512-character boundaries, the rep gets half a positioning statement. Recursive or document-structure chunking keeps the full argument intact.

[CITATION NEEDED — concept: chunking strategy impact on RAG retrieval in enablement workflows]

**Specific redirect**: This is the chunking decision that determines whether your Clay-enriched competitive intel surfaces complete or broken answers in a rep-facing search tool.

---

## Beat 5: Ship It (Production Considerations)

- **Chunk versioning**: when a source document updates, you must invalidate and regenerate affected chunks. Hash each chunk by source document ID + character offset.
- **Overlap cost**: 10% overlap on a 1M-token corpus means 100K extra tokens embedded and stored. Measure whether overlap improves recall in your specific retrieval task before shipping it.
- **Mixed strategies**: real corpora contain both structured docs (Markdown, HTML) and unstructured text (emails, call transcripts). Run document-structure chunking on structured sources and recursive chunking on unstructured sources. Do not use one strategy for everything.
- **Chunk size and context window**: chunks should fit within the context window minus space for the query and system prompt. For a 4K-token context window with a 500-token prompt and query, chunks should stay under 3K tokens with margin.

---

## Beat 6: Exercises

**Easy**: Run all four chunking scripts on the provided sample text. Print the chunk count and mean chunk length for each. Identify which strategy produces the most uniform chunk sizes.

**Medium**: Take a real Markdown document from your own knowledge base (competitive intel, ICP definition, playbook). Run document-structure chunking and recursive chunking on it. Print the first boundary from each. Write a one-sentence assessment of which boundary placement better preserves the argument.

**Hard**: Implement semantic chunking with a similarity threshold of 0.5 and 0.75 on the same document. Print chunk counts for both thresholds. Compare retrieval quality: embed 3 test queries, retrieve top-k chunks from each chunking result, and print whether the expected answer chunk appears in the top 3. Report which threshold, if either, improves recall over recursive chunking.
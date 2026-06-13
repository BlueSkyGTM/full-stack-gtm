# Question Answering Systems

## Hook
Practitioners have been sold "chat with your data" as a feature. This lesson breaks down what actually happens inside a retrieval-augmented QA pipeline — where it fails, where it doesn't, and what you can measure.

## Concept
The mechanism behind QA systems is a four-stage pipeline: chunk → embed → retrieve → generate. Each stage introduces a specific failure mode (bad splits, semantic drift, irrelevant retrieval, hallucinated answers). This beat covers dense passage retrieval, chunking boundaries, and the generation step that synthesizes retrieved context into an answer.

## Demo
Build a minimal QA system using an embedding model, a vector store, and an LLM. The code indexes a set of documents, accepts a natural language query, retrieves relevant chunks, and generates an answer with source attribution. All output is printed to terminal — no browser dependencies.

**Exercise hooks:**
- Easy: Modify the chunk size and observe retrieval quality change on the same query.
- Medium: Add a re-ranking step that scores retrieved chunks by cosine similarity threshold before passing to the LLM.
- Hard: Implement a hybrid retrieval strategy (keyword + semantic) and compare recall against semantic-only on a set of 10 test questions.

## Use It
**GTM Redirect: Zone 1 — ICP & Market Intelligence.** A QA system over your own account research notes, 10-K filings, and call transcripts is the mechanism behind "ask your knowledge base." This is not a chatbot — it is a retrieval pipeline that surfaces specific answers with citations. The Clay waterfall enriches accounts with firmographic data; the QA layer lets you query that enriched data in natural language.

**Exercise hooks:**
- Easy: Index a set of company 10-K excerpts and ask "What is [company]'s primary revenue model?"
- Medium: Build a QA pipeline over saved sales call transcripts that returns the answer with the transcript timestamp/source.
- Hard: Evaluate the QA system on 20 account-research questions where you already know the ground truth answer. Calculate exact match and F1.

## Ship It
Production QA systems require guardrails: source attribution must be enforced (not optional), retrieval latency must be measured, and hallucination detection must be implemented as a post-generation check. This beat covers deployment considerations: index refresh strategies, query logging, and the hard truth that most QA failures are chunking failures, not model failures.

**Exercise hooks:**
- Easy: Add a latency timer to each stage (chunk retrieval, LLM generation) and print a breakdown.
- Medium: Implement a confidence threshold — if no retrieved chunk exceeds a similarity score of 0.7, return "No relevant information found" instead of generating an answer.
- Hard: Build an evaluation harness that logs query, retrieved chunks, generated answer, and a manual quality score to a JSONL file for retrospective analysis.

## Evaluate
Assessment via `docs/en.md` objectives. Questions target the four-stage pipeline mechanism, chunking failure modes, retrieval quality metrics (recall@k, MRR), and the distinction between retrieval failure and generation failure.

---

**Learning Objectives (draft for `docs/en.md`):**
1. Implement a four-stage QA pipeline (chunk, embed, retrieve, generate) that answers questions over a document set with printed source attribution.
2. Diagnose retrieval failures versus generation failures by inspecting intermediate pipeline outputs.
3. Compare chunking strategies (fixed-size, sentence-boundary, semantic) and measure their impact on retrieval recall.
4. Evaluate QA system accuracy using exact match and F1 against a labeled test set.
5. Configure a confidence threshold on retrieval similarity scores to suppress hallucinated answers.
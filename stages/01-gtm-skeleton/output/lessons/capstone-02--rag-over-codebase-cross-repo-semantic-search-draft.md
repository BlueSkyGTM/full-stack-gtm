# Capstone 02 — RAG over Codebase (Cross-Repo Semantic Search)

## Hook

You're a solutions engineer who needs to answer "how do other customers implement our SDK?" across 47 repositories. Keyword search fails because naming conventions differ. You need semantic search over code — chunks, embeddings, and retrieval that respects code structure, not prose paragraphs.

## Concept

Explain chunking strategies specific to source code (function-level, class-level, file-level) and why token-window chunking from prose RAG breaks on code. Describe the embedding-then-retrieval pipeline: AST-aware splitting → embedding → vector store → query → re-ranking → context assembly → LLM generation. Contrast with naive chunking. Introduce the cross-repo challenge: disambiguating identical function names across repos, maintaining provenance metadata (repo, path, language, commit SHA) alongside each embedding.

## Build

Build the full pipeline end-to-end: (1) clone multiple sample repos, (2) parse and chunk code using tree-sitter or regex-based function extraction, (3) attach metadata (repo name, file path, language), (4) embed with a code-optimized model, (5) store in a vector database, (6) query with a natural-language question, (7) return ranked code snippets with provenance. All code runs in terminal with printed output confirming each stage.

**Exercise hooks:**
- Easy: Modify the chunk size and observe how retrieval quality changes for a specific query.
- Medium: Add a metadata filter so queries only return results from a specified repository.
- Hard: Implement a re-ranking step that scores snippets by structural similarity (e.g., same function signature pattern) after vector retrieval.

## Use It

Cross-repo semantic search maps directly to **Solutions Engineering and Technical Account Management** in Zone 30. When a prospect asks "how do I integrate your API with Flask?" you query the codebase RAG, pull real implementations from existing customers (anonymized), and return working examples instead of documentation links. The mechanism is identical: embed → retrieve → generate with context.

**Exercise hook:** Write three query prompts a solutions engineer would actually ask, run them against your pipeline, and evaluate whether the retrieved snippets are relevant enough to paste into a customer email.

## Ship It

Deploy the pipeline as a CLI tool that indexes repos on a schedule and answers queries from stdin. Cover: persistent vector store (don't re-embed on every run), incremental indexing (only new/changed files), and output formatting (markdown with repo links). Discuss cold-start cost: embedding 47 repos at ~$0.10/1M tokens with `text-embedding-3-small`.

**Exercise hook:** Package the tool so a teammate can run `python search.py --query "how to handle webhook retries"` and get back ranked results without reading your code.

## Stretch

**Exercise hooks:**
- Add multi-language support (Python + TypeScript + Go) and verify that a query like "authentication middleware" returns results across all three.
- Replace the code-optimized embedding model with a general-purpose one and compare retrieval recall on 10 queries — quantify the degradation.
- Build a simple web interface (Flask or FastAPI) that exposes the search as an API endpoint, confirming it returns JSON with provenance metadata.
# Capstone 02 — RAG over Codebase (Cross-Repo Semantic Search)

## Learning Objectives

- Build a cross-repository code search pipeline that chunks source files at AST boundaries, embeds function- and class-level chunks, and retrieves semantically relevant snippets using cosine similarity.
- Compare function-level, class-level, and file-level chunking strategies and measure their effect on retrieval quality for specific query types.
- Implement provenance metadata (repo, path, language, line numbers) alongside each embedding to disambiguate identical function names across repositories.
- Evaluate retrieval results against solutions engineering query patterns and rank them by relevance for customer-facing use cases.
- Deploy a persistent, incrementally-indexed CLI search tool that avoids re-embedding unchanged files on subsequent runs.

## The Problem

You are a solutions engineer who needs to answer "how do other customers implement our SDK?" across 47 repositories. Keyword search fails because naming conventions differ — one team calls it `retry_webhook`, another calls it `_deliver_with_backoff`, and a third buries the logic inside a class method called `process`. You need to find code by meaning, not by string match.

The naive approach — treat code as prose, split it into 512-token windows, embed, and search — breaks for three reasons. First, token-window splitting cuts functions in half. The embedding for a chunk that starts mid-function and ends mid-next-function captures nothing useful. Second, code has structure that prose does not: imports, class hierarchies, docstrings, and scope all carry meaning that a flat window discards. Third, across multiple repositories, the same function name (`handle_webhook`, `authenticate`, `retry`) appears everywhere. Without provenance metadata — which repo, which file, which line — retrieval results are useless because you cannot tell the user where to look.

The production answer is a pipeline that respects code structure at every stage: parse with an Abstract Syntax Tree (AST), chunk at node boundaries, embed with a model trained on code (not prose), store with rich metadata, and retrieve with provenance attached to every result.

## The Concept

### Why Prose Chunking Breaks on Code

Prose RAG splits text by token count — typically 512 tokens with 50-token overlap. This works for paragraphs because semantic boundaries in prose are fuzzy. Code boundaries are not fuzzy. A function is a discrete unit. Splitting it mid-body produces a chunk that starts with an indented return statement and ends with the first three lines of the next function. The embedding for that chunk averages two unrelated concepts, and retrieval quality collapses.

AST-aware chunking solves this by parsing the source file into its syntax tree and splitting at node boundaries. For Python, the `ast` module identifies every function definition, class definition, and top-level statement. For TypeScript, Go, and Rust, tree-sitter provides the same capability through a unified interface. Each chunk is a complete syntactic unit — a full function, a full class, or a coherent block of top-level code.

### The Pipeline

The full pipeline has seven stages, and each stage's output feeds the next:

```mermaid
flowchart LR
    A[Clone Repos] --> B[AST Parse Files]
    B --> C[Extract Function/Class Chunks]
    C --> D[Attach Provenance Metadata]
    D --> E[Embed Each Chunk]
    E --> F[Store in Vector Index]
    F --> G[Query → Embed → Cosine Search]
    G --> H[Rank + Assemble Context]
    H --> I[Return Snippets with Citations]
```

### Embedding Model Choice Matters

General-purpose embedding models (OpenAI `text-embedding-3-small`, `all-MiniLM-L6-v2`) are trained predominantly on natural language. They encode code poorly because they treat `def`, `import`, and `self` as ordinary tokens rather than structural keywords. Code-optimized models — Voyage `voyage-code-3`, Nomic `nomic-embed-code`, or OpenAI `text-embedding-3-large` with code prompting — are fine-tuned on source code and produce embeddings where two functions with identical behavior but different names land close in vector space.

For this capstone, the pipeline supports any OpenAI-compatible embedding endpoint. If no API key is available, it falls back to a TF-IDF vectorizer that demonstrates the retrieval mechanics — the chunks, metadata, and cosine search all work identically, but semantic quality drops because TF-IDF matches terms, not meaning.

### The Cross-Repo Disambiguation Problem

When you search for "webhook retry logic," you may get five chunks named `retry` from five different repos. The embeddings rank them, but without metadata, the user cannot act on the results. Every chunk must carry: repo name, file path, language, start line, end line, and chunk type (function or class). This provenance metadata turns a list of scores into a list of actionable references — "look at `sdk-core/retry.py`, lines 4-17, function `retry_with_backoff`."

### Re-Ranking

Vector search returns candidates by embedding similarity. A re-ranker scores these candidates on additional signals: structural similarity (does the function signature match the query's implied pattern?), docstring relevance (does the docstring semantically match the query?), and file-level context (is this function in a file that imports modules mentioned in the query?). Re-ranking does not replace vector search — it re-sorts the top 20 results using signals that dense embeddings miss.

## Build It

### Step 1: Create Sample Repositories and Chunk Them

This block creates three sample repositories with realistic integration code, then chunks every Python file at AST boundaries. The `ast` module parses each file, and we extract function and class nodes as discrete chunks with full provenance metadata.

```python
import ast
import json
import os
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path("sample_repos")
BASE_DIR.mkdir(exist_ok=True)

repos = {
    "flask-integrations": {
        "webhook_handler.py": '''def handle_webhook(payload):
    """Process incoming webhook events from the API."""
    signature = payload.get("signature")
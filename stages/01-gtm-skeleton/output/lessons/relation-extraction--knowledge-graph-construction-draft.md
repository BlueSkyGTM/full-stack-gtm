# Relation Extraction & Knowledge Graph Construction

## Hook

Raw text contains relationships that keyword search and vector similarity cannot reach. When you read "Acme Corp acquired Beta Inc for $2B," you implicitly extract the triple (Acme Corp, acquired, Beta Inc). Your systems don't — until you build a pipeline that does. This beat frames why structured relation extraction unlocks a different category of GTM intelligence than entity detection alone.

## Concept

Three mechanisms in sequence:

1. **Triple extraction**: The subject-predicate-object unit. Relation classification takes a sentence with two marked entities and predicts the predicate from a fixed schema (or open-domain extraction with no schema). Approaches include rule-based pattern matching over dependency parses, supervised classifiers trained on labeled data, and LLM-based structured extraction via prompting.

2. **Graph construction**: Triples become nodes and edges. Entity resolution merges mentions ("Acme", "Acme Corp", "Acme Corporation") into canonical nodes. Adjacency list or property graph storage.

3. **Graph querying**: Traversal, path finding, and subgraph extraction over the constructed graph. This is where the extracted knowledge becomes retrievable.

Key tension: fixed schema (high precision, low recall) vs. open extraction (high recall, noisy). The practitioner picks based on downstream tolerance for false edges.

## Demonstration

Build a working pipeline in two stages:

- Stage 1: Use spaCy dependency parsing to extract relations from sentences with rule-based patterns over syntactic structure. Print each extracted triple with confidence flags.
- Stage 2: Feed the same sentences to an LLM prompt requesting structured JSON triples. Parse the output, build a NetworkX directed graph from both extraction methods, run a shortest-path query between two entities, and print the path.

All code runs in terminal. Observable output: printed triples, graph statistics, and query results.

## Guided Practice

- **Easy**: Modify the rule-based dependency patterns to capture a new relation type (e.g., "partners_with" via prepositional attachments). Print extracted triples to verify.
- **Medium**: Implement entity resolution by normalizing entity surface forms to canonical IDs before graph insertion. Print before/after node counts to show deduplication.
- **Hard**: Build an evaluation function that compares extracted triples against a small gold-standard set. Compute precision, recall, and F1. Print the scores and the false-positive triples.

## Use It

GTM Redirect: **Zone 3 — Enrichment**. Relation extraction feeds the enrichment layer of a GTM stack. When Clay (or any enrichment tool) pulls unstructured data — news articles, SEC filings, job postings, press releases — relation extraction converts that text into structured triples that populate account and contact records. This is how you build a "competitive landscape" graph (acquisitions, partnerships, leadership changes) or a "technology stack" graph (company → uses_tool → tool_name) from raw web content. The knowledge graph becomes a queryable enrichment source that answers questions vector search alone cannot: "Which accounts in my TAM recently acquired a company in the cybersecurity space?"

## Ship It

Production deployment checklist for a relation extraction pipeline:

- **Deduplication**: Merge duplicate triples across extraction runs. Hash on (canonical_subject, predicate, canonical_object).
- **Confidence scoring**: Attach a confidence value to each triple. Threshold based on downstream precision requirements.
- **Incremental updates**: Don't rebuild the graph from scratch. Append new triples and re-resolve entities.
- **Storage**: NetworkX for prototyping. Neo4j or similar property graph database for production scale.
- **Monitoring**: Track extraction volume, novel-triple rate, and entity-resolution merge rate per run. A sudden drop in novel triples signals a source problem.

Ship exercise: wrap the pipeline in a CLI tool that accepts a directory of text files, extracts triples, builds/updates a persisted graph, and prints a summary report. Run it twice on the same data to confirm idempotent deduplication.
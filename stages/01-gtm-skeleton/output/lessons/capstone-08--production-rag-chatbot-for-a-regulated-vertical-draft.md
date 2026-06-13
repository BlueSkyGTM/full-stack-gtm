# Capstone 08 — Production RAG Chatbot for a Regulated Vertical

## Set It

Define the constraints that separate a regulated-vertical RAG system from a general-purpose chatbot: per-document access controls, mandatory citation, audit trails, and content-policy enforcement. Present the architecture diagram showing ingestion, retrieval with permission filtering, generation with source attribution, and logging.

## See It

Run a working end-to-end RAG pipeline against a corpus of 10 mock regulatory documents (e.g., simplified HIPAA guidance sheets). The pipeline retrieves, filters by a simulated user role, generates an answer with inline citations, and writes an audit log entry to a local JSON file. Observable output: printed answer with bracketed source references, and a confirmation line showing the audit record was written.

## Study It

Examine four mechanisms in sequence: (1) metadata-tagged chunk ingestion with role-based access fields, (2) retrieval-time post-filtering that removes chunks the user's role cannot access, (3) prompt construction that forces the model to cite source IDs or respond "not available in permitted sources," and (4) append-only audit logging of every retrieval set and final output. Compare this to a naive RAG pipeline that retrieves and generates without any of these layers — show where the naive version fails a compliance check.

## Solve It

Three exercise tiers. **Easy**: Add a second user role and verify that retrieval filtering returns different chunk sets for each role on the same query. **Medium**: Implement a content-policy guard that flags or blocks generated answers containing specific regulated phrases (e.g., medical advice disclaimers). **Hard**: Build a citation-fidelity evaluator that checks whether every bracketed source ID in the answer actually corresponds to a chunk in the retrieval set, and report precision/recall of citations.

## Use It

This RAG-with-citations architecture maps directly to GTM Zone 1 (ICP Research & Enrichment) when building account research tools that must attribute every claim to a source — for example, a sales engineer who queries earned media mentions, 10-K filings, and press releases and needs sourced answers for buyer conversations. The same retrieval-filtering mechanism that enforces role-based access in regulated RAG is the mechanism behind Clay's data-credential waterfall, where enrichment stops at the first provider that returns a permitted result.

## Ship It

Package the pipeline as a CLI tool that accepts a user ID (mapped to a role), a query string, and a flag for audit-log destination. On execution, it prints the sourced answer and writes the audit record. The deliverable is the tool plus a test script that runs three queries across two roles and asserts correct filtering and citation fidelity. Include a brief write-up (3–5 paragraphs) explaining which regulatory constraints the system satisfies, which it does not, and what would be required to close the gaps.

---

### Learning Objectives

1. Implement a RAG pipeline with document-level access controls that filters retrieval results by user role before generation.
2. Configure append-only audit logging that captures the query, retrieval set, user role, and final generated output.
3. Build a citation-fidelity evaluator that verifies every source reference in a generated answer maps to a retrieved chunk.
4. Deploy a CLI-based chatbot that enforces content-policy rules on generated output for a regulated vertical.
5. Compare a compliant RAG architecture against a naive RAG pipeline and identify specific failure modes in each compliance dimension.
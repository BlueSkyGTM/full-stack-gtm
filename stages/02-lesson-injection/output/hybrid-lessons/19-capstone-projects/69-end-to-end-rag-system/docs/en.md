# End-to-End RAG System

## Learning Objectives

- Compose a chunker, hybrid retriever, cross-encoder reranker, and citation-aware answer generator into a single end-to-end RAG pipeline that runs from raw documents to grounded answers.
- Implement answer generation that cites retrieved chunks by source file and passage index, with refuse-on-low-confidence fallback when retrieval scores fall below threshold.
- Evaluate the integrated pipeline against stage-isolated baselines on a fixture query set, measuring recall@k and answer faithfulness to prove composition beats isolation.
- Diagnose pipeline failure modes —
# ColPali and Vision-Native Document RAG

## GTM Redirect Rules

- GTM cluster: **Zone 1 — Account & ICP Intelligence** (document parsing for account research, SEC filings, pitch decks)
- Specific redirect: "vision-native retrieval over account research documents" when processing PDFs that resist OCR-based chunking (tables, charts, multi-column layouts)
- Foundational fallback applies for architectural concepts with no direct GTM mapping

---

## Beat 1: Hook It

Show two retrieval results over the same messy PDF — one from a chunked-text pipeline (OCR → markdown → embed) and one from ColPali's image-native approach. The text pipeline misses a key table; ColPali retrieves it. Pose the question: what if the document *is* the index?

## Beat 2: Ground It

Explain the failure mode of traditional document RAG: OCR quality, reading order ambiguity, table/chart information loss, and chunking boundaries that split visual entities. Then introduce the mechanism shift — encode the rendered page as an image, let a vision model produce token-level embeddings, and match queries against visual patches using late interaction (MaxSim). Reference ColBERT as the textual ancestor of this scoring pattern.

## Beat 3: Explain It

Walk through the ColPali architecture: (1) document page rendered to image, (2) ViT patch tokenizer produces patch embeddings, (3) projection to a shared embedding space, (4) query encoded by the same model's text encoder, (5) MaxSim late interaction scoring between every query token and every visual patch. Explain why multi-vector representation preserves spatial information that single-vector embedding destroys. Note: ColPali builds on PaliGemma weights; behavior of the projection layer is documented in the paper but implementation details around patch resolution should be verified against the source.

## Beat 4: Build It

Exercise hooks only:

- **Easy**: Load ColPali from `vidore/colpali` via `transformers`, encode a single PDF page rendered as a PIL Image, print the shape of the multi-vector embedding.
- **Medium**: Encode 5 document pages and 3 queries, compute MaxSim scores manually between each query and each page, print the ranked retrieval results.
- **Hard**: Build a minimal document retriever that accepts a PDF, renders pages to images, encodes them, and returns top-k pages for a natural language query — with timing output comparing indexing latency to a naive `sentence-transformers` text baseline.

## Beat 5: Use It

Redirect: **Zone 1 — Account & ICP Intelligence**, specifically document parsing for account research. When enriching account records, GTM teams often need to extract signal from SEC filings, earnings decks, and product PDFs that contain tables and charts. Traditional OCR pipelines lose table structure. ColPali retrieves the relevant *page* visually, then the page can be passed to a VLM for extraction. This is not "use ColPali for everything" — it is "use ColPali when your source documents resist clean text extraction." Compare against `docs/en.md` objectives to confirm alignment.

## Beat 6: Ship It

Production concerns: GPU memory for multi-vector storage (each page produces ~1000+ patch embeddings), approximate nearest neighbor index options for multi-vector retrieval [CITATION NEEDED — concept: ANN indexes for multi-vector / late-interaction embeddings in production], latency profile of MaxSim scoring at scale, and the decision boundary for when text-based RAG is sufficient (clean prose documents) vs. when vision-native is justified (complex layouts, tables, figures). Ship threshold: benchmark retrieval accuracy on your actual document corpus before committing to the infrastructure cost.
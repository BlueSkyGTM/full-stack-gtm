# Document and Diagram Understanding

## Hook (Beat 1)
Documents and diagrams are the dark matter of GTM data — contracts in PDFs, architecture diagrams in security reviews, org charts in seller attachments. Raw text extraction misses structure; raw vision misses semantics. This lesson covers the mechanism that bridges both: multimodal document understanding where layout, visual elements, and text are processed as a unified signal.

## Concept (Beat 2)
Three-layer document cognition. Layer 1: OCR and text extraction (Tesseract, cloud OCR APIs) — gets characters, loses spatial relationships. Layer 2: Layout analysis — bounding boxes, reading order, table detection, figure/section segmentation. Layer 3: Semantic fusion — a vision-language model consumes the rendered page image plus extracted text to produce grounded interpretations. Diagrams require the same stack but with a visual-reasoning emphasis: detecting nodes, edges, labels, and spatial relationships in flowcharts, architecture diagrams, and org charts. The key mechanism is that the model doesn't read — it *sees* and *reads simultaneously*, which is why a table extracted as plain text loses meaning but a table processed as an image retains it.

## Code (Beat 3)
A working Claude API call that takes a document image (PNG/PDF page), extracts structured data from both text regions and a diagram, and prints parsed output — table rows, diagram node relationships, and labeled elements. Observable output confirms extraction worked. Second example: same document processed with OCR-only (e.g., `pytesseract`) to show what's lost without the vision channel.

## Use It (Beat 4)
GTM redirect: **Zone 2 — Enrichment**. Inbound RFPs, security questionnaires, and vendor assessment documents arrive as PDFs with mixed content — tables of compliance controls, architecture diagrams, and prose sections. A Clay enrichment waterfall [CITATION NEEDED — concept: Clay document processing in enrichment waterfall] can invoke a document-understanding step to extract structured fields from these attachments, populating account or deal records without manual data entry. The specific mechanism: uploaded document → vision-language extraction → structured field mapping → write to record.

## Ship It (Beat 5)
Production considerations: cost per page (vision tokens vs. OCR API calls), latency budgets for real-time vs. batch, fallback chains (try multimodal extraction → fall back to OCR + heuristic parsing → flag for human review), and accuracy validation loops. Exercise hooks:
- **Easy**: Process a single-page PDF, print extracted table and diagram nodes.
- **Medium**: Build a two-stage pipeline — OCR first, then vision model on flagged regions — and compare cost/accuracy.
- **Hard**: Implement an accuracy audit: run 20 documents through the pipeline, auto-compare extracted fields against a ground-truth JSON, and log precision/recall per field type.

## Quiz Hook (Beat 6)
Questions target the mechanism, not trivia:
- Why does OCR-only extraction fail on tables that vision-language processing handles correctly?
- Given a diagram with nodes A→B→C where B is unlabeled, what does the model infer and what information source enables that inference?
- Name the fallback chain order and justify why multimodal comes before OCR, not after.
- A Clay enrichment step processes a vendor security PDF. What three layers of document cognition must occur before structured field mapping?
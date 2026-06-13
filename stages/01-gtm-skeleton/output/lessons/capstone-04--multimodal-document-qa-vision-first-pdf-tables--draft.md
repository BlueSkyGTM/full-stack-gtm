# Capstone 04 — Multimodal Document QA (Vision-First PDF, Tables, Charts)

## GTM Redirect Rules

This capstone maps to the **Intent Data & Enrichment** cluster (Zone 1). Vision-first document QA extracts structured intelligence from PDFs that text-only pipelines miss: pricing tables in RFPs, org charts in pitch decks, market sizing charts in analyst reports. The redirect: "this is the extraction layer that feeds enrichment workflows in Clay and similar tools — when text-only parsing returns garbage from a table, vision-first recovers the structure."

---

## Hook

Most B2B intelligence lives in PDFs nobody reads fully: security questionnaires, RFPs, competitor battlecards with embedded charts. Text-only extraction (PyPDF2, pdfplumber) silently fails on tables and returns gibberish from charts. Vision-first QA treats each page as an image and asks a multimodal model directly — table structure preserved, chart semantics extracted. This capstone builds a pipeline that takes any PDF and answers questions against its full visual content.

---

## Concept Introduce

**Two extraction paradigms exist for PDFs:** text-native extraction (parse the text layer) and vision-first extraction (render pages to images, send to a vision-language model). Text-native is fast and cheap but fails on scanned documents, tables with merged cells, and any chart. Vision-first is slower and more expensive but recovers layout semantics that text parsers cannot.

The mechanism: render each PDF page to a PNG, send each image to a vision-capable LLM with a structured extraction prompt, and merge results into a unified document representation for downstream QA. For tables specifically, instruct the model to return markdown or JSON. For charts, instruct it to return the underlying data points it can infer from visual inspection.

Hybrid approach: use text extraction first (cheap, fast), flag pages where extraction confidence is low or where structural elements are detected, then fall back to vision-only on those pages. This is the cost-optimized production pattern.

---

## Concept Demo

**Demo 1: Text vs. Vision extraction on a table-heavy page.** Build a comparison: extract text from a sample PDF page using `pdfplumber`, then extract the same page using vision (render to image, send to Claude). Print both outputs side-by-side to show what text parsing misses.

**Demo 2: Chart data extraction.** Send a rendered page containing a bar chart or pie chart to Claude with a prompt asking for the underlying data as JSON. Print the recovered data structure.

**Demo 3: Full pipeline — PDF to QA.** Process a multi-page PDF through: (1) render all pages to images, (2) extract structured content per page with vision, (3) build a combined context, (4) answer a natural-language question against the full document. Print question and answer with source page citation.

All code runs via Claude Code Desktop with `anthropic` SDK. Uses `pypdfium2` or `pdf2image` for rendering. No browser dependency.

---

## Use It

**GTM application: RFP/Security Questionnaire parsing for deal desk automation.**

When a prospect sends a 40-page security questionnaire or RFP, a text-only parser will fail on the compliance matrices (tables), network architecture diagrams, and pricing grids. Vision-first QA lets you:

1. Extract every table as structured JSON (feed into Clay enrichment columns)
2. Pull data points from charts (market sizing, competitive positioning)
3. Answer "does this prospect's requirements match our capabilities?" against the full visual document

This is the extraction mechanism that makes document-based enrichment reliable. In Clay specifically, this pattern would be implemented as an external enrichment step (via webhook or API integration) that returns structured data back into Clay rows. [CITATION NEEDED — concept: Clay webhook integration pattern for custom enrichment]

**Exercise hooks:**
- *Easy:* Process a single-page PDF with one table, extract to JSON, print result.
- *Medium:* Build the hybrid pipeline — text extraction first, vision fallback on flagged pages — and compare cost/accuracy.
- *Hard:* Feed extracted document content into a RAG-style retrieval system and answer 5 questions about the document with source citations.

---

## Ship It

**Production deliverable: A CLI tool that takes a PDF path and a question, returns an answer with page-level citations and extracted tables/charts as structured data.**

Requirements:
- Accept any PDF via file path
- Render pages to images using `pypdfium2` (no system dependencies beyond pip packages)
- Process each page with vision extraction (tables → JSON, charts → data points, text → markdown)
- Merge extractions into a unified document context
- Accept a natural-language question and return answer with `source_page` field
- Output structured JSON to stdout
- Handle multi-page documents (10+ pages) without token overflow by chunking context intelligently

The CLI tool must run unmodified and produce observable JSON output. Include a sample PDF generation step (create a test PDF with code that includes a table and text) so the exercise is self-contained.

---

## Evaluate

**Assessment criteria (testable against learning objectives):**

1. **Detect when vision-first extraction is necessary.** Given outputs from text extraction and vision extraction on the same page, identify which correctly recovers a merged-cell table. (Observable: print both, human verifies, or automated diff against ground truth.)

2. **Implement page rendering to images.** Code that takes a PDF and produces PNGs for each page, with printed confirmation of page count and image dimensions.

3. **Extract tables and charts as structured data.** Prompt engineering that consistently returns valid JSON from table images and data arrays from chart images. Validated by attempting `json.loads()` on the output.

4. **Build a cost-optimized hybrid pipeline.** Justify which pages get vision treatment vs. text-only. Print a per-page cost estimate and total.

5. **Answer questions with source attribution.** Given a multi-page document, answer questions and correctly cite which page(s) the answer came from. Verified against the actual document content.

---

**Learning Objectives (3–5, action verbs only):**

1. Compare text-native vs. vision-first extraction on table-heavy and chart-heavy PDF pages, identifying failure modes of each approach.
2. Implement a PDF-to-image rendering pipeline that produces page-level images suitable for vision model ingestion.
3. Configure vision model prompts that extract tables to JSON and chart data to structured arrays with consistent schema.
4. Build a hybrid extraction pipeline that routes pages to text or vision processing based on structural detection.
5. Construct a document QA system that answers natural-language questions with page-level source citations across multi-page PDFs.
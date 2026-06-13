# OCR & Document Understanding

## Hook

You have 500 scanned vendor invoices. Each one has a company name, total amount, date, and line items. Manual re-entry is 30 seconds per invoice — over 4 hours of tedium. OCR converts image pixels into text; document understanding converts that text into structured fields you can write to a database. This lesson covers both.

## Concept

**The OCR pipeline has four stages:** image preprocessing, text region detection, character recognition, and post-processing correction. Classic engines like Tesseract use pattern matching on character glyphs. Modern document understanding models (LayoutLM, Donut, cloud APIs from AWS/Google/Azure) add layout awareness — they learn that "Total: $4,200" at the bottom-right of an invoice means something different than "$4,200" appearing mid-paragraph. Layout context is what separates raw text extraction from usable structured data.

Key mechanisms:
- **Preprocessing:** grayscale conversion, binarization (thresholding), deskew, noise removal — each improves recognition accuracy on degraded scans
- **Text detection vs. recognition:** detection finds bounding boxes; recognition classifies pixels within those boxes as characters
- **Layout analysis:** column detection, reading order, table structure, form field grouping
- **Structured extraction:** regex, rule-based parsers, or LLM-based extractors that turn raw OCR text into JSON with named fields

## Demo

Code examples (all runnable in terminal):

1. **Basic OCR with Tesseract** — load an image, run `pytesseract.image_to_string()`, print extracted text, measure confidence scores
2. **Preprocessing comparison** — run OCR on the same image before and after grayscale + threshold + deskew; print character accuracy delta
3. **Layout-aware extraction** — use `pytesseract.image_to_data()` to get bounding boxes and text; reconstruct reading order; identify form fields by spatial position
4. **Structured output** — take OCR text, extract invoice fields (vendor, date, total) using regex patterns, output as JSON

## Use It

**GTM redirect: Zone 3 — Enrichment pipelines.**

Inbound documents (signed contracts, vendor invoices, scanned business cards, PDF purchase orders) contain account and contact data that needs to reach your CRM. OCR + document understanding is the bridge between "a file arrived in a mailbox" and "a record was created or updated in Salesforce/HubSpot."

Application: Build a document intake webhook that accepts a PDF, runs OCR, extracts company name + contact email + purchase amount, and writes the result as a structured payload — ready for a Clay enrichment waterfall or direct CRM API call.

[CITATION NEEDED — concept: GTM enrichment pipeline for inbound document processing]

## Ship It

- **Easy:** Run Tesseract OCR on three provided test images (clean scan, rotated scan, low-contrast photo). Print the raw text output and confidence scores for each. Report which preprocessing step would fix each failure mode.

- **Medium:** Build an invoice parser that takes a scanned invoice image, extracts vendor name, invoice date, and total amount into a JSON object, and prints the result. Handle at least two layout variations.

- **Hard:** Build a document classifier + extraction pipeline. Given a directory of mixed documents (invoices, business cards, contracts), classify each document type, route to the appropriate field extraction logic, and output a single JSON array with all extracted records. Track and print accuracy against provided ground truth.

## Assess

Questions target: OCR pipeline stage identification, preprocessing technique selection for specific image degradation types, layout-aware vs. raw OCR trade-offs, structured extraction pattern matching, failure mode diagnosis (low confidence scores, character substitution patterns, reading order errors).

---

**Learning Objectives:**

1. Implement OCR text extraction from images using Tesseract with configurable preprocessing
2. Compare OCR output quality across preprocessing techniques (grayscale, binarization, deskew)
3. Extract structured fields from OCR text using bounding box layout data and pattern matching
4. Build a document-to-JSON pipeline that handles at least two document layout variations
5. Diagnose OCR failure modes from confidence scores and character-level output
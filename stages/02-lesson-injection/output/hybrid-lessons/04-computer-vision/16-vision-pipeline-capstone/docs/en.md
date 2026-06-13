# Build a Complete Vision Pipeline — Capstone

## Learning Objectives

- Wire ingest, preprocess, infer, and postprocess stages into a single executable pipeline with fixed data contracts between each stage
- Apply confidence thresholds in postprocessing to trade recall for precision and observe the effect on filtered output counts
- Emit structured JSON from the pipeline that downstream systems — APIs, databases, GTM enrichment tools — can consume without parsing raw model tensors
- Map the four-stage vision pipeline architecture onto a GTM enrichment waterfall to recognize the shared Find → Enrich → Transform → Export pattern

## The Problem

You have used individual vision components in isolation. A classifier predicts a label. A detector draws boxes. A segmenter fills masks. Each one works on its own bench. Production vision systems do not work that way — they chain these components into a single pipeline that takes raw image bytes through preprocessing, inference, postprocessing, and structured output. That chain is the same architecture whether you are auditing retail shelves, pre-screening medical images, or classifying prospect logos in a GTM workflow.

Wiring those chains is the part that separates an ML prototype from a product. Every interface between stages is a new place for silent failures: a tensor shaped wrong, a normalization applied twice, a confidence threshold set on logits instead of probabilities. The model itself is rarely the bottleneck — the interfaces between stages are where production bugs live. A retail shelf audit is a detector plus a product classifier plus a price-OCR step. Each handoff between those models is a coordinate transform, a resize, a format conversion. If any of those contracts is implicit or undocumented, the pipeline breaks the moment someone swaps a component.

This capstone wires the minimum viable pipeline
# InternVL3: Native Multimodal Pretraining

## Hook

The standard pipeline for vision-language models trains a text LLM, freezes it, then bolts on a vision encoder through a thin projection layer. That bottleneck limits what the model can do with visual information — the language backbone never truly "sees." InternVL3 questions that assumption by training the LLM parameters jointly with visual inputs from the start, making the model natively multimodal rather than retrofitted.

---

## Concept

InternVL3 replaces the "train language first, attach vision later" pipeline with a single native multimodal pretraining stage. The core mechanism: LLM weights update on interleaved image-text data during pretraining, rather than staying frozen during visual alignment. This produces a model where visual and textual representations share the same representational space at the parameter level, not just at a projection layer. The result is improved performance on document understanding, chart reading, OCR, and multi-image reasoning — tasks where the "bolted-on" approach loses signal at the connector.

---

## Mechanism

Three architectural components and one training decision:

1. **InternViT-6B** serves as the vision encoder, producing visual tokens from input images.
2. **A pixel-shuffle + MLP projector** downsamples and maps visual tokens into the LLM's embedding space, reducing token count while preserving spatial information.
3. **InternLM3 (or comparable LLM backbone)** processes the combined visual and text tokens — but unlike prior InternVL versions, its weights are **not frozen** during the visual pretraining phase.

The training pipeline runs in progressive stages: (a) native multimodal pretraining on large-scale interleaved image-text corpora with LLM weights unfrozen, (b) supervised fine-tuning on high-quality multimodal instruction data, and (c) preference optimization (DPO or similar). The critical difference from InternVL2 is step (a) — prior work froze the LLM during initial alignment and only unfroze it during supervised fine-tuning.

[CITATION NEEDED — concept: InternVL3 training stage details and data mixture proportions from the original paper]

---

## Use It

**GTM Redirect: Zone B — Enrichment, multimodal data extraction pipeline.**

When enriching account or contact records, some signal lives in images: screenshots of a prospect's website, logos, pitch deck PDFs rendered to images, social media profile screenshots. A natively multimodal model can extract structured fields (company name, tagline, tech stack indicators, branding colors, team size from team page screenshots) from these visual inputs without the information loss that occurs when a projection-only model compresses visual tokens into a frozen language space.

In a Clay enrichment workflow, this would replace or augment a text-only enrichment step: pass a company website screenshot to InternVL3, prompt for specific structured fields, and write the extracted data back to the record. The "native" training matters here because document-style images (dense text in visual layout) are exactly the domain where bolted-on vision models degrade — the LLM backbone hasn't learned to route visual-textual features through its own layers.

[CITATION NEEDED — concept: specific Clay waterfall configuration for multimodal enrichment steps]

---

## Ship It

Exercise hooks (no full exercise text):

**Easy:** Load a pretrained InternVL3 checkpoint (via HuggingFace transformers or the official InternVL inference repo), pass a single image of a business card, and print extracted name, title, company, and email. Confirm the model handles OCR-level extraction without a separate OCR engine.

**Medium:** Build a batch inference script that processes a directory of website screenshots, extracts a fixed schema (company_name, tagline, primary_cta_text, tech_stack_hints), and writes results to a JSONL file. Time the inference and compare output quality against a non-native multimodal model (e.g., a projection-only VLM) on the same inputs.

**Hard:** Implement a lightweight enrichment pipeline that takes a list of company domains, screenshots their homepages via a headless browser, runs InternVL3 inference to extract structured fields, and outputs a CSV suitable for import into a CRM. Measure extraction accuracy against human-labeled ground truth for 20 companies.

---

## Evaluate It

1. **Compare and explain:** What is the architectural difference between InternVL2 and InternVL3's approach to the LLM backbone during visual pretraining, and what capability does this difference unlock?

2. **Diagnose:** Given a scenario where InternVL3 fails to extract text from a dense table image but succeeds on a single-line business card, identify whether the failure is in the vision encoder, projector, or LLM backbone — and justify your reasoning based on the native pretraining mechanism.

3. **Implement:** Write inference code that loads InternVL3, passes two images (a logo and a team page screenshot), and prompts the model to answer a question requiring cross-image reasoning (e.g., "Does the team page show more than 10 people?"). Print the model's response and token-level confidence if available.

4. **Evaluate tradeoffs:** A GTM team wants to use InternVL3 for real-time enrichment during sales calls (processing screen shares). Estimate the minimum hardware requirements based on model size (InternViT-6B + InternLM-8B), and propose whether native multimodal pretraining is justified over a smaller projection-only model given latency constraints. State your assumptions.
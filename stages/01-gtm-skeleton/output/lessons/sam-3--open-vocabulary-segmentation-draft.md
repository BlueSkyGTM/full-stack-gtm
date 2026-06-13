# SAM 3 & Open-Vocabulary Segmentation

## GTM Redirect Rules
Open-vocabulary segmentation maps to visual content analysis in GTM — specifically, competitive screenshot analysis, brand asset parsing, and ad creative decomposition. If the connection is thin for a given subsection, the redirect defaults to "foundational for Zone 3 (enrichment pipelines)."

---

## Beat 1: Hook — The Mask Without a Label

You can segment every object in an image perfectly and still know nothing. SAM produces masks; it does not produce meaning. Open-vocabulary segmentation closes that gap by pairing mask geometry with free-text semantic grounding. Describe why mask-only output breaks downstream when you need to act on *what* was segmented, not just *that* something was segmented.

---

## Beat 2: Concept — Geometry Meets Semantics

Explain the two-stage mechanism: (1) a mask proposal backbone generates candidate regions without class constraints, and (2) a vision-language aligner scores each region against arbitrary text prompts. Compare this to closed-vocabulary approaches where the class list is frozen at training time. Cover how SAM-family models handle the first stage, and how CLIP-style text-image embeddings handle the second. Note: as of current knowledge, "SAM 3" is not a released model designation — if the user means SAM 2.1 or a community checkpoint, flag that ambiguity here rather than assuming. `[CITATION NEEDED — concept: SAM 3 official release and architecture changes from SAM 2.1]`

---

## Beat 3: Demo — Segment and Classify

Run a working Python script that loads an image, generates masks with a SAM-family checkpoint, then classifies each mask against a user-supplied list of text labels using CLIP similarity scoring. Print the top label and confidence for each detected region. All code runs in terminal, prints observable output, no browser dependency.

**Exercise hooks:**
- *Easy:* Modify the text label list and re-run on the same image.
- *Medium:* Threshold the confidence score and only print masks above 0.5.
- *Hard:* Replace the hardcoded image with one fetched from a URL, and add a timing report for mask generation vs. text scoring.

---

## Beat 4: Use It — Screenshot Decomposition for Competitive Intel

Map open-vocabulary segmentation to GTM enrichment: feed a competitor's landing page screenshot into the pipeline, segment into regions (hero, CTA, pricing table, testimonial), and classify each region by function. This is the visual analogue of DOM parsing — useful when you don't have DOM access or are analyzing ad creatives where no DOM exists.

**Exercise hooks:**
- *Easy:* Run the demo pipeline on a screenshot and identify which region is most likely a CTA.
- *Medium:* Output a JSON structure with region label, bounding box, and confidence for integration into an enrichment tool.
- *Hard:* Batch-process 10 screenshots and aggregate which UI components appear most frequently across competitors.

---

## Beat 5: Ship It — Latency, Batching, and Model Selection

Cover the engineering constraints: SAM mask generation is GPU-bound and latency-sensitive. CLIP scoring is cheap per region but scales with mask count. Discuss filtering strategies (suppress small masks, limit proposals) and model choice (SAM 2.1 vs. MobileSAM vs. FastSAM) based on throughput requirements. Address checkpoint storage, ONNX export for inference, and batch processing patterns.

**Exercise hooks:**
- *Medium:* Time the full pipeline on the same image with two different SAM checkpoints and print a comparison table.
- *Hard:* Export the pipeline to ONNX, run inference, and compare latency against the PyTorch path.

---

## Beat 6: Debug It — Phantom Masks and Misaligned Labels

Catalog failure modes: (1) oversegmentation producing dozens of meaningless small masks, (2) CLIP text similarity flattening across ambiguous regions, (3) mask boundaries that don't align with semantic object boundaries. Show how to diagnose each by inspecting mask area distributions, similarity score spreads, and overlay visualizations printed as ASCII or saved to disk.

**Exercise hooks:**
- *Easy:* Add a filter that drops any mask below 2% of total image area and print the before/after count.
- *Medium:* For each mask, print the top-3 text labels and their score gaps — flag masks where the gap between #1 and #2 is below 0.05.
- *Hard:* Generate a text report that flags which masks have high IoU overlap with other masks, indicating likely oversegmentation.

---

## Learning Objectives (draft)

1. Implement a two-stage segmentation pipeline that combines mask proposals with open-vocabulary text classification.
2. Compare closed-vocabulary and open-vocabulary segmentation in terms of flexibility, latency, and label coverage.
3. Configure mask filtering thresholds to suppress noise in high-mask-count outputs.
4. Evaluate segmentation quality by inspecting score distributions, mask area distributions, and inter-mask overlap.
5. Deploy the pipeline on batch screenshot data and extract structured JSON describing UI component regions.
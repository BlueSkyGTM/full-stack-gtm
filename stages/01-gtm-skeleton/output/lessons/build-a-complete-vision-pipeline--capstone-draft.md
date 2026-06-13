# Build a Complete Vision Pipeline — Capstone

## Hook (Beat 1)
You've used individual vision components in isolation. Now you wire them into a single pipeline that takes raw image input through preprocessing, inference, post-processing, and structured output — the same architecture production systems use.

## Concept (Beat 2)
A complete vision pipeline chains four stages: **ingest** (load and validate images), **preprocess** (resize, normalize, augment to match model expectations), **infer** (run the model forward pass), and **postprocess** (apply NMS, filter by confidence, format results). Each stage has a fixed contract — array in, array out — so stages are swappable without rewriting neighbors. The pipeline is the unit of deployment, not the model.

## Demo (Beat 3)
Build a working pipeline that processes a directory of images through a pre-trained classifier, applies confidence filtering, and writes structured JSON results. observable output: printed summary of detections with counts and flagged images.

## Use It (Beat 4)
This pipeline pattern maps directly to the **enrichment waterfall** in GTM systems — specifically, processing prospect company logos, screenshots, or visual assets to classify vertical or extract intent signals. [CITATION NEEDED — concept: vision-based ICP enrichment using logo/brand detection in GTM workflows] The pipeline contract (input → transform → score → filter) is structurally identical to how Clay sequences enrichment steps.

## Ship It (Beat 5)
Three exercises:
- **Easy**: Add a new postprocessing filter that removes results below a configurable confidence threshold and prints the before/after count.
- **Medium**: Extend the pipeline with a second model stage (e.g., object detection after classification) and merge the outputs into a single JSON structure.
- **Hard**: Add error handling for corrupt/missing images, implement a retry mechanism for failed inference calls, and write a manifest file logging every image's pipeline status (success/failure/skipped).

## Review (Beat 6)
The pipeline pattern — ingest, preprocess, infer, postprocess — is the deployment unit. Each stage's input/output contract is what makes it maintainable. Confidence thresholds in postprocessing are where you trade recall for precision. This same waterfall structure appears in GTM enrichment sequences where multiple data providers are called in fallback order.
# Open-Vocabulary Vision — CLIP

## Hook

CLIP maps images and text into the same embedding space, enabling zero-shot classification without retraining. This means you can describe any visual concept in plain language and find matching images — no labeled dataset required.

## Concept

Contrastive Language-Image Pre-training trains a dual-encoder (vision + text) on 400M image-text pairs by maximizing cosine similarity for matched pairs and minimizing it for negatives. The result: a shared embedding space where "a photo of a dog" lives near actual dog photos. The mechanism is contrastive loss over mini-batches, not generative modeling.

## Demonstration

Load a pre-trained CLIP model, encode an image alongside a set of candidate text prompts, compute cosine similarities, and return the highest-scoring label. Print the ranking to confirm the zero-shot prediction worked.

*Exercise hook (Easy):* Classify a single image against 3 hand-written prompts. Print similarity scores.
*Exercise hook (Medium):* Build a function that takes a directory of images and a label taxonomy, then outputs a classification CSV.
*Exercise hook (Hard):* Implement the contrastive loss function from scratch on a toy batch of embeddings and verify the gradient direction matches the library implementation.

## Use It

Zero-shot visual classification maps directly to **enrichment waterfall** stages in GTM — specifically, classifying company website screenshots, logo styles, or product imagery into a taxonomy without training custom classifiers. The redirect: when your Clay waterfall needs to categorize visual assets at scale, CLIP replaces a supervised image classifier that would require hundreds of labeled examples per category. [CITATION NEEDED — concept: CLIP used in enrichment waterfall for visual company data]

## Ship It

Production deployment of CLIP requires batching image preprocesses (resize + center-crop + normalize to 224×224), managing text prompt templates for consistent scoring, and thresholding on softmax temperature. The model runs inference on GPU; CPU inference is viable but slow at ~200ms per image.

*Exercise hook (Medium):* Write a batch inference pipeline that processes 50 images with a shared label set and prints per-image latency.

## Review

CLIP's mechanism — contrastive dual-encoding — produces a shared embedding space that generalizes to unseen categories. The tradeoff: performance degrades on domain-specific imagery underrepresented in web-scraped training data (e.g., medical imaging, satellite). If the visual concept isn't described often in internet text, CLIP's zero-shot accuracy drops.

---

**Learning Objectives:**
1. Implement zero-shot image classification by computing cosine similarity between CLIP image and text embeddings.
2. Compare contrastive loss behavior across matched and unmatched image-text pairs.
3. Configure text prompt templates to improve classification accuracy on a custom label taxonomy.
4. Evaluate CLIP's zero-shot performance degradation on out-of-distribution image domains.
5. Build a batch inference pipeline for classifying multiple images against a shared label set.
# CLIP and Contrastive Vision-Language Pretraining

## Learning Objectives

1. Implement the InfoNCE contrastive loss function over paired image-text embeddings.
2. Compute and interpret cosine similarity matrices between image and text embedding batches.
3. Evaluate zero-shot classification accuracy using CLIP on a labeled image set.
4. Configure CLIP-based visual classification for a GTM enrichment task (logo detection, page-type tagging).

---

## Beat 1: Hook

The data your enrichment pipelines ignore: every prospect has a website with screenshots, logos, and product imagery that your text-only pipeline treats as invisible. CLIP makes visual signal computationally accessible — not by tagging images with labels, but by placing images and text into the same vector space where similarity is just a dot product.

---

## Beat 2: Concept

**The contrastive mechanism.** You have a batch of N image-text pairs. The image encoder produces N embeddings. The text encoder produces N embeddings. The correct pairing is the diagonal — image i goes with text i. InfoNCE loss pulls those diagonal pairs together in cosine space while pushing every off-diagonal combination apart. The temperature parameter controls how sharply the model penalizes mismatched pairs. This is the entire training signal: no bounding boxes, no class labels, no object detection heads. Just "these belong together, those don't."

**Dual-encoder architecture.** ViT or ResNet for images. Transformer for text. Both project into a shared D-dimensional space. No cross-attention between modalities at training time — the alignment happens purely through the contrastive loss landscape.

**Zero-shot transfer.** At inference, you classify an image by encoding it alongside a set of candidate text prompts ("a photo of a dog", "a photo of a cat") and picking the highest cosine similarity. No fine-tuning required. The model learned the alignment during pretraining on 400M image-text pairs scraped from the web.

---

## Beat 3: Demo

Load a pretrained CLIP model (OpenAI's `ViT-B/32` via `transformers` or `openai/clip` weights). Encode a batch of 4 images and 4 captions. Compute the 4×4 cosine similarity matrix. Show the diagonal dominance — the matched pairs score higher than the off-diagonal. Then show what happens when you shuffle the captions: the diagonal breaks. This is the contrastive signal made visible.

Code produces a printed similarity matrix. No plotting, no notebook — terminal output only.

---

## Beat 4: Lab

**Exercise hooks:**

- **Easy:** Given a precomputed similarity matrix, identify which text caption best matches each image by finding the argmax per row. Print the matches.
- **Medium:** Implement InfoNCE loss from scratch using PyTorch. Given a batch of image embeddings and text embeddings (both normalized), compute the cross-entropy loss where the target is the diagonal index. Print the loss value and verify it decreases over 3 gradient steps on random data.
- **Hard:** Build a zero-shot classifier for 5 categories using hand-written text prompts. Encode a set of test images, compute similarities to all category prompts, and print classification accuracy. Then modify one prompt and observe the accuracy change.

---

## Beat 5: Use It

**GTM Redirect: Zone 1 Enrichment — Visual Signal Extraction.**

[CITATION NEEDED — concept: CLIP-based company logo detection in GTM enrichment pipelines]

The practical application: you have a CSV of company domains. You screenshot each homepage (or pull favicon/logo URLs). You run CLIP to classify each image against prompts like "a technology company logo", "a healthcare company logo", "a blank page", "a login screen", "an e-commerce storefront". The result is a new column in your enrichment table: `page_type` or `logo_category`, derived entirely from visual signal with zero custom training.

This is the same mechanism Clay uses when enriching accounts — visual classification of company assets without requiring human-labeled training data for each new taxonomy.

---

## Beat 6: Ship It

**Exercise hook (hard):** Write a CLI script that accepts a CSV of domains, downloads the first image from each URL (or uses a local directory of screenshots), runs CLIP zero-shot classification against a user-defined set of category prompts passed as a JSON config, and writes the enriched CSV with a new `visual_category` column and a confidence score. Print a summary of classification distribution to stdout.

**GTM Redirect:** This pipeline outputs enrichment data in the format Clay's waterfall consumes — a column of categorical predictions with confidence scores, ready to merge into your account scoring model. The visual dimension is the input; the structured category is the output. Zone 1 enrichment, implemented.

---

## Notes

- CLIP's training data (400M image-text pairs from the web) introduces biases. Zero-shot accuracy is not uniform across domains. Industrial/technical imagery often underperforms consumer imagery. Test before relying on it for vertical-specific GTM tasks.
- The temperature parameter in contrastive loss is learned during CLIP training. At inference, it's fixed. If you see overconfident similarity scores on your domain, logit-scale calibration may be necessary.
- No fine-tuning is demonstrated here. Fine-tuning CLIP on domain-specific pairs requires curated image-text data and careful learning rate scheduling — out of scope for this lesson but the logical next step for production GTM deployments.
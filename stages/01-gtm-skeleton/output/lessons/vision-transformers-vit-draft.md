# Vision Transformers (ViT)

## Beat 1: Hook

Images are grids of pixels. Convolutions exploit local structure. But what if you treat an image as a sequence—like a sentence—and apply the same self-attention that powers GPT? That's ViT. In GTM, this matters the moment you need to extract structured data from screenshots, logos, or document scans where OCR alone fails.

## Beat 2: Concept

**Core mechanism:** An image is split into fixed-size patches (e.g., 16×16), each patch is linearly projected into a token, positional embeddings are added, and the resulting sequence passes through a standard transformer encoder. A `[CLS]` token aggregates global information for classification. No convolutions, no inductive bias for locality—just attention over spatial tokens.

**Key tradeoff:** ViT needs massive pre-training data to learn spatial relationships that CNNs get for free. With insufficient data, ViT underperforms ResNet. With sufficient data, ViT overtakes.

**Architecture variants to name:** ViT-B/16, ViT-L/14, ViT-H/14 (Base/Large/Huge, patch size as divisor of 224).

**Exercise hooks:**
- Easy: Patch an image manually and count tokens for a given resolution and patch size.
- Medium: Implement the patch embedding step from scratch (linear projection of flattened patches).
- Hard: Build the full forward pass—patching, position embedding, transformer encoder—using only `torch.nn.TransformerEncoder`.

## Beat 3: Use It

**GTM redirect:** Zone 02 (Enrichment) — visual document understanding for company intelligence. Specifically: using ViT-based models to classify or extract structured information from company documents, logos, or screenshots where the visual layout matters (not just text content).

**Mechanism in context:** A ViT model processes a screenshot of a company's pricing page. The `[CLS]` token output feeds into a classifier that detects pricing model (freemium, usage-based, enterprise-only). This is enrichment via visual understanding, not text scraping.

**Tool:** Hugging Face `transformers` library — `ViTForImageClassification` and `ViTFeatureExtractor` implement the patch-to-token pipeline, position embeddings, and encoder stack. You load a pretrained checkpoint and run inference.

**Exercise hooks:**
- Easy: Load a pretrained ViT from Hugging Face and classify a single image. Print predicted class and confidence.
- Medium: Extract the `[CLS]` token embedding from ViT and compute cosine similarity between two company logos to detect if they share visual properties.
- Hard: Build a small classifier head on top of frozen ViT `[CLS]` embeddings using a labeled dataset of pricing page screenshots.

## Beat 4: Build It

Implement a working ViT inference pipeline. Load a pretrained checkpoint (`google/vit-base-patch16-224`), preprocess an image, run forward pass, decode predictions. All code runs in terminal, outputs class predictions with confidence scores.

**Exercise hooks:**
- Easy: Classify a single image with ViT and print top-5 predictions.
- Medium: Batch-process a directory of images and write results (filename, top prediction, confidence) to CSV.
- Hard: Fine-tune ViT on a custom dataset (e.g., logo vs. no-logo classification) using Hugging Face `Trainer` API. Print validation accuracy after each epoch.

## Beat 5: Ship It

**Production considerations:** ViT-B/16 inference at 224×224 takes ~10ms on GPU, ~100ms on CPU. Batch inference amortizes overhead. For GTM enrichment pipelines processing thousands of company screenshots, throughput matters more than single-image latency.

**Optimization path:** Export to ONNX, then to TensorRT or OpenVINO for inference acceleration. Quantization (INT8) cuts latency ~2x with <1% accuracy drop for classification tasks.

**Deployment pattern:** Wrap ViT inference in a FastAPI endpoint. Accept image URL or base64 payload. Return structured classification output. Queue-based processing for batch enrichment jobs.

**Exercise hooks:**
- Easy: Export a ViT model to ONNX and verify output parity with PyTorch (max diff < 1e-5).
- Medium: Build a FastAPI endpoint that accepts an image URL, runs ViT inference, returns JSON with predicted class and confidence.
- Hard: Implement batch inference with dynamic batching (accumulate requests for 50ms, batch forward pass, return individual results). Print throughput metric.

## Beat 6: Extend It

**Data hunger problem:** ViT trained on ImageNet-1K (1.3M images) underperforms ResNet. ViT trained on ImageNet-21K (14M) or JFT-300M matches or exceeds. If you're fine-tuning on a small GTM dataset (hundreds of company screenshots), start from a checkpoint pretrained on 21K or larger.

**Variants worth knowing:**
- **DeiT:** Data-efficient training via distillation from a CNN teacher. Makes ViT viable with ImageNet-1K scale data.
- **Swin Transformer:** Hierarchical attention with shifted windows. Better for dense prediction tasks (detection, segmentation) than vanilla ViT.
- **DINO / DINOv2:** Self-supervised ViT training. Produces strong visual features without labeled data. Relevant when you have lots of unlabeled company images.

**Multi-modal extension:** CLIP (ViT image encoder + text encoder, contrastive loss). Enables zero-shot image classification using text descriptions. Directly applicable to GTM: classify company screenshots using custom text labels without training a classifier.

**Exercise hooks:**
- Easy: Load DINOv2 ViT features for two images and compute cosine similarity. Print result.
- Medium: Use CLIP (ViT-based) to zero-shot classify a set of images against custom text labels (e.g., ["pricing page", "about page", "blog post", "contact page"]). Print predictions.
- Hard: Compare ViT-B/16 vs. DeiT-base vs. Swin-base on a small classification task (your choice of dataset). Print accuracy, inference time, and model size for each.

---

**Learning Objectives:**
1. Implement the patch embedding pipeline that converts an image into a sequence of tokens for transformer processing.
2. Compare ViT and CNN approaches in terms of data requirements, inductive bias, and scaling behavior.
3. Deploy a pretrained ViT model for image classification using Hugging Face `transformers`.
4. Extract and use ViT `[CLS]` token embeddings as feature representations for downstream tasks.
5. Evaluate when ViT is the appropriate architecture choice versus convolutional alternatives for a given visual task.
# Image Classification

## Learning Objectives

1. Classify images using a pre-trained convolutional neural network and extract predicted labels with confidence scores.
2. Compare feature extraction vs. fine-tuning as two transfer learning strategies.
3. Evaluate prediction confidence thresholds to filter low-certainty classifications.
4. Implement a logo-detection classifier for company enrichment in a GTM pipeline.
5. Diagnose misclassifications by inspecting per-class probability distributions.

---

## Beat 1: Hook

Every screenshot, logo, and document attachment sitting in your CRM is an unclassified signal. Image classification turns pixels into labels — and labels into routing decisions. This lesson covers the mechanism that makes that possible: convolutional feature extraction followed by a linear classifier head.

---

## Beat 2: Concept

A convolutional neural network stacks three operations: convolution (local pattern detection), pooling (spatial downsampling), and linear projection (class scoring). Pre-trained models like ResNet50 have already learned edge, texture, and shape detectors from millions of images. Transfer learning replaces the final classification head while keeping those frozen feature detectors — the mechanism that lets a general-purpose model specialize with minimal new data.

---

## Beat 3: Code

Load a pre-trained ResNet50 from `torchvision`, run inference on a sample image, and print the top-5 predicted classes with confidence scores. Observable output: a ranked list of label–probability pairs.

**Exercise hooks:**
- (Easy) Swap the input image and confirm the top prediction changes.
- (Medium) Add a confidence threshold filter that suppresses predictions below 0.10.
- (Hard) Replace ResNet50 with EfficientNet-B0 and compare the top-5 overlap.

---

## Beat 4: Use It

**GTM cluster: Zone 1 — Enrichment / Signal Capture.** Company logo detection is a concrete image-classification task: crawl a prospect's website, classify the hero image or favicon, and route high-confidence matches into the correct ICP bucket. This is the classification-then-routing pattern used in automated enrichment pipelines — classify the visual asset, then branch the workflow based on the label.

**Exercise hooks:**
- (Easy) Run inference on a saved screenshot of a company homepage and print the detected brand category.
- (Medium) Build a function that accepts a directory of screenshots, classifies each, and outputs a CSV of filename → predicted label → confidence.

---

## Beat 5: Ship It

Production image classifiers fail on distribution shift — screenshots look different from training data, and confidence scores compress near 0.5. Calibrate with a held-out validation set from your actual domain (not ImageNet). Log per-class confidence distributions over time; when they drift, retraining is due. For low-latency enrichment, export the model to ONNX and serve via a lightweight REST endpoint — avoid loading PyTorch at inference time.

**Exercise hooks:**
- (Easy) Add a timing wrapper around inference and print mean latency over 10 runs.
- (Hard) Export the model to ONNX, run inference with `onnxruntime`, and compare output parity with the PyTorch version.

---

## Beat 6: Assess

Questions target the mechanism, not trivia:

1. Explain why freezing convolutional layers preserves feature detectors during transfer learning.
2. Given a per-class probability vector, identify which prediction should be rejected and why.
3. Compare the trade-offs of feature extraction vs. fine-tuning when the target domain has only 200 labeled images.

**Exercise hooks:**
- (Medium) Write a function that accepts a probability vector and returns "uncertain" if the gap between the top two classes is below a configurable threshold — print the decision for 5 test vectors.
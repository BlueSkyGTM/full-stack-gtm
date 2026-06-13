# Video Understanding — Temporal Modeling

## Learning Objectives

- Compare the four temporal modeling families (2D+pool, 3D convolution, two-stream, video transformer) by compute cost, receptive field, and whether they learn motion explicitly
- Implement frame sampling, temporal pooling, and three video classifier architectures in PyTorch
- Trace how temporal resolution and temporal receptive field jointly determine which actions a model can recognize
- Build a conversation analysis pipeline that applies temporal segmentation to detect objection moments and compute talk-to-listen ratios
- Evaluate whether a given GTM dataset has a temporal axis that warrants sequence modeling or whether static classification suffices

## The Problem

A 30-second video at 30 fps is 900 images. Naively, video classification is image classification run 900 times followed by some kind of aggregation. That approach works when the action is visible in almost every frame — sports, cooking, exercise videos — and fails badly when the action is defined by motion itself. "Pushing something from left to right" from the Something-Something dataset looks like two nearly identical still objects in consecutive frames. The histogram difference between frame *t* and frame *t+1* is negligible. The meaning lives entirely in the delta.

Consider a person picking up a phone versus putting it down. Frame-by-frame, the two actions produce visually similar images — a hand near a phone. A single-frame classifier sees "hand holding phone" in both cases. Only the temporal ordering of frames disambiguates "lifting" from "lowering." This is why temporal modeling matters: the direction of change across frames carries the semantic content that any single frame cannot.

The core question for every video architecture is: **when does temporal structure get modelled, and how?** The answer drives compute cost, pretraining strategy, whether ImageNet weights transfer, and which actions the model can even represent. A model that pools frame-level predictions at the end (2D+pool) cannot distinguish "open drawer" from "close drawer." A model with 3D convolutional kernels or temporal attention can — if it processes enough frames at sufficient resolution. This lesson covers the architectural families that answer this question and shows how the same temporal modeling mechanisms power sales call analysis platforms that detect objection moments and sentiment shifts across recorded conversations.

## The Concept

Four architectural families solve temporal modeling, each making different assumptions about how motion information enters the model.

**3D Convolution (C3D, I3D).** The mechanism extends 2D convolution kernels from spatial dimensions (H, W) to spatiotemporal dimensions (T, H, W). A single 3D kernel slides across both space and time, learning features like "object moving leftward" in one operation. I3D ("Inflated 3D ConvNet") takes this further by literally inflating 2D ImageNet-pretrained kernels into 3D — a 3×3 2D filter becomes a 3×3×3 3D filter with weights copied and normalized. This is why I3D transfers well: it starts from proven ImageNet features rather than training from scratch. The limitation is compute: each kernel processes O(T × H × W × C) elements, and the temporal receptive field grows only with kernel size and network depth. A 3×3×3 kernel at the first layer sees 3 frames; stacking layers widens the field but slowly.

**Two-Stream Networks (Simonyan & Zisserman, 2014).** Instead of forcing the network to discover motion from raw RGB, this architecture feeds motion explicitly as optical flow — a precomputed field of per-pixel displacement vectors between consecutive frames. One stream processes RGB frames (appearance), another processes stacked optical flow (motion), and late fusion combines predictions. The mechanism is clean: explicit motion signal as input. The limitation is that optical flow computation is expensive (often the bottleneck at inference time) and flow is a handcrafted representation that may discard information the network could have learned itself [CITATION NEEDED — concept: two-stream architecture original paper benchmarks on UCF101].

**Video Transformers (TimeSformer, ViViT).** These architectures apply self-attention across time positions, giving any frame direct access to any other frame in a single layer. TimeSformer decomposes full spatiotemporal attention into separate spatial-only and temporal-only passes to reduce the O(T² × H × W) cost. ViViT embeds 3D "tubelets" — patches that span both space and time — as input tokens. The mechanism is global temporal context: frame 1 can attend to frame 300 without intermediate layers. The limitation is O(T²) memory scaling, which forces chunking
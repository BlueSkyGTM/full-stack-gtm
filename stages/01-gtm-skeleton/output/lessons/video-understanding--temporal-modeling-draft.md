# Video Understanding — Temporal Modeling

## GTM Redirect Rules

- **GTM Cluster:** Signal Processing — Rich Media (Zone 1)
- **Redirect:** Temporal modeling is the mechanism behind sales call analysis platforms (Gong, Chorus) that detect objection moments, sentiment shifts, and talk-to-listen ratios across time. The same sequence modeling that detects actions in video detects buying signals in recorded conversations.
- **Foundational note:** If the application is content scoring of static assets (no temporal dimension), the redirect is "foundational for Zone 1" — temporal modeling only applies when the GTM data has a time axis.

---

## Beat 1: Hook

Video is not a bag of frames. A person picking up a phone vs. putting it down produce nearly identical frame histograms — the meaning lives in *what changes across time*. Temporal modeling is the difference between classifying snapshots and understanding events. Every sales call analysis tool, every action detection system, every "highlights" extractor depends on this mechanism.

---

## Beat 2: Concept

**Mechanism: Temporal modeling captures dependencies across frames.**

Three architectural families solve this, each with tradeoffs:

1. **3D Convolution (C3D, I3D).** Extends 2D convolution kernels from (H, W) to (T, H, W). The kernel learns spatiotemporal features in a single operation. Mechanism: local temporal receptive field grows with kernel size and depth. Limitation: fixed receptive field, expensive compute — O(T × H × W × C) per kernel.

2. **Two-Stream Networks (Simonyan & Zisserman, 2014).** Separate networks for RGB frames (appearance) and optical flow (motion). Late fusion combines both. Mechanism: explicit motion signal as input rather than learned from raw frames. Limitation: optical flow computation is expensive; flow is a handcrafted representation. [CITATION NEEDED — concept: two-stream architecture original paper benchmarks]

3. **Temporal Attention / Video Transformers (TimeSformer, ViViT).** Apply self-attention across time positions. TimeSformer decomposes attention into spatial-only and temporal-only for efficiency. ViViT embeds 3D "tubelets" as tokens. Mechanism: global temporal context in one layer — any frame attends to any other frame. Limitation: O(T²) memory; long videos require chunking.

4. **Sequential Models (LSTM/GRU on frame features).** Extract per-frame features with a 2D CNN, then run a recurrent model across the sequence. Mechanism: hidden state accumulates temporal information. Limitation: sequential processing prevents parallelization; vanishing gradients over long sequences.

**Key distinction:** Temporal resolution vs. temporal receptive field. A model processing 1 frame/second has low temporal resolution but can still have a large receptive field (processing 300 seconds at 1 fps). A model processing 30 fps has high resolution but may only see 10 seconds. Action recognition generally needs resolution ≥ the duration of the atomic action. [CITATION NEEDED — concept: Kinetics-400 benchmark temporal resolution requirements]

---

## Beat 3: Demonstration

Implement a minimal temporal processor that demonstrates the core mechanisms without requiring a GPU or downloading large weights:

1. **Frame-level feature extraction:** Use a pretrained 2D CNN (MobileNetV3 or ResNet18 from torchvision) to extract per-frame feature vectors from a synthetic video sequence (generated programmatically — colored shapes moving across frames).

2. **Temporal aggregation — compare three methods:**
   - **Mean pooling across time** (no temporal modeling — baseline)
   - **1D temporal convolution** (local temporal patterns)
   - **Simple self-attention across time** (global temporal patterns)

3. **Observable output:** Print classification accuracy for each method on a synthetic action detection task (e.g., "object moving left" vs. "object moving right" vs. "object stationary"). The temporal models should outperform mean pooling, confirming that temporal information matters.

All code runs in terminal. Synthetic video is generated with PIL/numpy — no camera or download needed.

**Exercise hooks:**
- *Easy:* Modify the motion pattern in the synthetic video generator (change speed, add diagonal movement). Observe how accuracy changes.
- *Medium:* Replace the 1D temporal convolution with a bidirectional GRU. Compare accuracy and training time.
- *Hard:* Implement TimeSformer-style decomposed attention (spatial attention within each frame, then temporal attention across frames). Measure memory usage vs. full attention.

---

## Beat 4: Use It

**GTM Application: Sales Call Temporal Analysis**

The same temporal modeling that classifies actions in video classifies *moments* in recorded sales calls. A Gong or Chorus processes call recordings as time-series: audio features extracted per segment, then temporal models detect objection moments, competitor mentions, and engagement shifts.

**Mechanism mapping:**
- **Per-frame features** → per-segment audio/text features (speaker embedding, sentiment score, keyword presence)
- **Temporal modeling** → detecting that "price objection" typically follows "feature demo" and precedes "discount discussion"
- **Action localization** → "at timestamp 14:32, the prospect expressed concern about integration" — this is temporal action detection applied to conversational signals

**Exercise hook:**
- *Medium:* Given a synthetic transcript with timestamped sentiment scores, implement a 1D convolution to detect "objection patterns" (sentiment drop + competitor mention within a 30-second window). This is the same mechanism as action detection, applied to 1D conversational signals instead of 2D video features.

---

## Beat 5: Ship It

**Production considerations for temporal models:**

1. **Memory scaling.** Self-attention is O(T²). For a 60-minute sales call at 1 segment/second, that's 3,600² = ~13M attention pairs. Chunking is not optional — it is architectural. Common pattern: process in overlapping windows (e.g., 60-second windows, 10-second overlap), then aggregate window-level predictions.

2. **Inference latency.** 3D convolution requires loading T frames simultaneously. For real-time processing (live call analysis), use causal temporal models (only attend to past frames) or streaming architectures. For batch processing (post-call analysis), latency is less constrained.

3. **Temporal stride tradeoff.** Processing every frame is expensive. Stride-4 (process every 4th frame) reduces compute by 4× but may miss sub-stride events. Empirical guideline: stride should be ≤ half the duration of the shortest event you need to detect. [CITATION NEEDED — concept: temporal stride benchmarks for action detection]

4. **Evaluation.** Use mean Average Precision (mAP) at temporal IoU thresholds for action detection tasks. For classification tasks (whole-video labels), top-1 and top-5 accuracy are standard. Report both — they measure different things.

**Exercise hook:**
- *Hard:* Take the demonstration model and benchmark it at temporal strides of 1, 2, 4, 8. Plot accuracy vs. stride. Identify the stride at which accuracy degrades below 90% of the stride-1 baseline. This is your deployment stride.

---

## Beat 6: Evaluate

**Assessment hooks (not full quiz text — just stems for quiz generation):**

1. **Mechanism identification:** Given a description of a model that "processes optical flow frames in one network and RGB frames in another, then averages predictions," identify the architectural family and explain *why* optical flow is used instead of raw frame differencing.

2. **Tradeoff analysis:** Explain why a 3D convolution with kernel size (3, 7, 7) has a smaller temporal receptive field than a self-attention layer, and describe a concrete scenario where this difference affects detection accuracy.

3. **Deployment decision:** A GTM team wants to detect "competitor mention" moments in 90-minute sales calls in real-time. Recommend an architecture (from the four families covered) and justify: temporal resolution needed, memory constraints, latency requirements.

4. **Code tracing:** Given the demonstration code (mean pooling vs. 1D conv vs. self-attention), predict which method will fail on a synthetic video where the action is "object alternates direction every 5 frames" — explain why.

---

## Learning Objectives

1. **Compare** the four temporal modeling architectures (3D conv, two-stream, temporal attention, recurrent) on receptive field, computational cost, and parallelizability.
2. **Implement** temporal aggregation over frame-level features using 1D convolution and self-attention, and **measure** accuracy against a mean-pooling baseline.
3. **Configure** temporal stride for a given action detection task based on the minimum event duration.
4. **Map** video temporal modeling mechanisms to sales call analysis pipelines — specifically, frame-level features → segment-level features, action detection → objection moment detection.
5. **Evaluate** a temporal model's suitability for real-time vs. batch processing given memory and latency constraints.
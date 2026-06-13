# Monocular Depth & Geometry Estimation

---

## 1. Hook

You have a single RGB image. No stereo rig, no LiDAR, no structured light. Your task: recover the 3D geometry that produced that 2D projection. This is an ill-posed inverse problem — infinite 3D scenes can produce the same 2D image. Monocular depth estimation networks learn strong priors over scene structure to pick the most plausible reconstruction. The output is never "correct" in a geometric sense, but it is often useful enough.

---

## 2. Concept

**The ill-posed problem.** A single image collapses a 3D scene along the depth axis. Recovering depth from one view requires assumptions about the world — ground planes are horizontal, objects sit on surfaces, distant things are above near things in the image.

**Relative vs. metric depth.** Most monocular networks output *relative* or *ordinal* depth — "this pixel is farther than that pixel." Metric depth (absolute distance in meters) requires either metric training data or a known reference scale in the scene. Confusing the two is the most common integration mistake.

**Scale ambiguity.** A monocular network cannot distinguish between a small object close to the camera and a large object far away without external scale cues. This is not a model limitation — it is a geometric impossibility. Any system consuming monocular depth must either accept relative output or provide a scale reference.

**Training paradigms:**
- **Supervised:** Trained on paired (image, ground-truth depth) datasets from LiDAR or structured light (KITTI, NYU Depth V2). Produces metric depth within the training domain.
- **Self-supervised (stereo):** Trained on stereo pairs where the network learns to reconstruct one view from the other using predicted depth. No ground-truth depth needed. Produces relative depth up to an unknown baseline scale.
- **Self-supervised (video):** Trained on video sequences using photometric consistency across time and camera ego-motion. Scale drift is a known failure mode.

**Architecture families:**
- **Encoder-decoder with skip connections** (MiDaS): Multi-scale feature extraction, progressive upsampling. Strong generalization across domains.
- **Dense Prediction Transformer (DPT)** (Depth Anything, DPT-Hybrid): Vision transformer encoder, reassemble tokens into dense prediction. Better global context, higher compute cost.
- **Adaptive bins** (AdaBins): Predicts depth as a mixture of bin centers, adapts bin boundaries per image. Better at handling diverse depth ranges.

**Geometry beyond depth.** Surface normals can be derived from depth via finite differences. Plane estimation can follow via RANSAC on the depth map. These derived quantities are what downstream systems actually consume for scene understanding.

**Confidence and failure modes.** Monocular depth networks fail silently on:
- Transparent/reflective surfaces (glass, mirrors)
- Thin structures (wires, poles)
- Textureless regions
- Out-of-distribution scenes (aerial views when trained on ground-level)

No monocular depth estimator provides reliable per-pixel confidence. Treat every output as uncalibrated unless you validate on your specific domain.

---

## 3. Implement

**Exercise Easy:** Load a pretrained MiDaS or Depth Anything model. Run inference on a local image. Render the depth map as a grayscale heatmap and print min/max/mean depth values to confirm the network produced output.

**Exercise Medium:** Compute surface normals from the predicted depth map using finite differences. Render the normal map as an RGB image where each channel encodes one normal component. Print the fraction of normals with magnitude < 0.1 to detect degenerate regions.

**Exercise Hard:** Implement a minimal scale-recovery pipeline. Given a monocular depth map and one known real-world distance in the scene (provided as a pixel pair + ground-truth distance), recover the scale factor and produce a metric depth map. Validate against a second known distance and report the relative error.

---

## 4. Use It

**GTM redirect:** This is foundational for Zone 5 (Content & Creative). Monocular depth estimation enables 3D-aware analysis of 2D marketing assets — product photography composition, scene layering in ad creative, and spatial reasoning about visual content. If you are building enrichment pipelines that ingest company imagery or product photos, depth maps provide a geometric signal that 2D classifiers cannot.

**Specific mechanism:** A pipeline that ranks product photos by "visual depth complexity" (variance in predicted depth, number of distinct depth layers) can surface which images have strong spatial composition vs. flat, unengaging shots. This is a heuristic, not a law — but it is a measurable, automatable heuristic.

---

## 5. Ship It

**Latency and memory.** MiDaS-small runs at ~30ms per 384×384 image on a T4 GPU. Depth Anything-large runs at ~150ms per 518×518 image. Model size ranges from 30MB (MiDaS-small) to 400MB (Depth Anything-large). Choose based on your throughput budget, not your accuracy aspirations — monocular depth is inherently approximate, and the accuracy gap between small and large models is smaller than the gap between any model and ground truth.

**Batch processing.** If you are scoring an image corpus (product catalog, ad creative library), batch inference is straightforward — stack images, run the model, write depth statistics to a database. The depth map itself is rarely worth storing; derived statistics (depth variance, number of depth layers, dominant plane orientation) are the features downstream systems consume.

**Domain shift.** Models trained on indoor scenes (NYU) perform poorly outdoors and vice versa. If your images come from a specific domain (e.g., aerial drone shots, medical imaging, microscopic product photos), validate on a held-out set before trusting any output. The model will produce a depth map regardless — it will just be wrong.

**No confidence signal.** Monocular depth networks do not emit reliable uncertainty estimates. If you need to know when the model is wrong, you need an external validation mechanism (comparison to a known 3D model, multi-view consistency check, or human review of flagged outputs).

---

## 6. Evaluate

- Compare the output of relative vs. metric depth modes on the same image. Identify which is appropriate for a given task and justify the choice.
- Given a depth map with known failure regions (provided), compute surface normals and identify which normal regions are corrupted by the depth errors. Explain the propagation mechanism.
- Evaluate two pretrained models (e.g., MiDaS-small vs. Depth Anything-small) on a held-out domain. Report which performs better and whether the difference justifies the compute cost delta.
- Implement the scale-recovery pipeline from the hard exercise. Given two reference distances, quantify the error when using only one for calibration vs. both.
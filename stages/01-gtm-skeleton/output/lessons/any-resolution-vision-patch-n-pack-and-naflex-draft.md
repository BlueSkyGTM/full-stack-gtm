# Any-Resolution Vision: Patch-n'-Pack and NaFlex

## Outline

---

### Beat 1: Concept

Fixed-resolution inputs are a legacy constraint from CNN architectures that ViTs inherited uncritically. Patch-n'-Pack (NaViT) and NaFlex break this by treating vision as a sequence packing problem: patches from multiple images at multiple resolutions share a single forward pass. Describe the shift from "resize everything to 224×224" to "pack tokens until you hit the context budget." Mention computational overhead of naive variable-resolution batching and why packing solves it.

---

### Beat 2: Mechanism

Explain the three mechanisms in order:

1. **Patch extraction at native resolution** — images are split into patches of fixed pixel size (e.g., 16×16), but the *number* of patches varies per image depending on its dimensions. No resizing, no cropping, no distortion.

2. **Token packing** — patches from multiple images are concatenated into a single sequence with attention masks that prevent cross-image attention. This is the same pattern as packing multiple documents into one Transformer sequence (used in LLM training). Describe the attention mask structure: block-diagonal, where each image's patches can only attend to patches from the same image.

3. **Positional encoding for variable-length sequences** — standard learned positional embeddings assume a fixed grid. NaViT uses factorized positional embeddings (separate row and column embeddings) or continuous positional encodings that generalize to unseen resolutions. [CITATION NEEDED — concept: NaFlex positional encoding strategy, whether it differs from NaViT's factorized approach]

Note where the mechanism is well-documented (NaViT packing, attention masking) and where it is not (NaFlex's specific architectural modifications, if any, beyond NaViT).

---

### Beat 3: Code

**Exercise hook (easy):** Write code that takes three images of different resolutions, extracts 16×16 patches from each, packs them into a single token sequence with an attention mask, and prints the sequence shape and mask structure.

**Exercise hook (medium):** Implement factorized positional embeddings (row index + column index → embedding) and verify they produce correct embeddings for patches from a 480×640 image and a 224×224 image in the same packed sequence.

**Exercise hook (hard):** Build a minimal packing scheduler that takes a list of image resolutions and a max token budget, assigns images to batches using a first-fit-decreasing bin-packing heuristic on patch counts, and prints the batch assignments with utilization percentage.

All code runs in terminal. No display dependencies. Output is printed shapes, mask values, batch assignments.

---

### Beat 4: Use It

**GTM Redirect:** This is foundational for Zone 2 enrichment pipelines that process visual signals — company logos, product screenshots, ad creatives, document scans — at native resolution without information loss from downscaling. In Clay workflows, any enrichment step that feeds images into a vision model benefits from variable-resolution processing when the input quality varies (high-res product photos vs. low-res favicon thumbnails).

**Exercise hook (medium):** Given a directory of company logos at varying resolutions (simulated as arrays of different shapes), write a patch counter that logs each logo's patch count and flags logos that would exceed a 1024-token budget individually. Print a summary table: filename, resolution, patch count, flag.

---

### Beat 5: Ship It

**Exercise hook (hard):** Build a batch preparation module that accepts a list of image file paths, extracts patches at their native resolution, groups them into batches using the packing scheduler from Beat 3, and writes each batch to disk as a `.pt` file containing `{tokens, attention_mask, position_ids, image_boundaries}`. Include a validation function that loads each batch and confirms: (1) total tokens ≤ budget, (2) attention mask is block-diagonal, (3) no image is split across batches. Print pass/fail for each check.

**GTM Redirect:** This is the preprocessing layer for any production pipeline that runs vision models over heterogeneous image collections — enrichment workflows, competitive screenshot monitoring, creative asset analysis.

---

### Beat 6: Evaluate

Three to five questions grounded in mechanism, not trivia:

1. Given two images (320×480 and 640×640) with patch size 16, compute patch counts and determine whether they fit in a single 2048-token packed sequence.
2. Draw the attention mask structure for a packed sequence containing two images. Mark which positions are allowed to attend to each other.
3. Explain why standard learned positional embeddings (fixed grid, e.g., 14×14) fail when an image produces a 20×30 grid of patches. Describe one solution.
4. A packing scheduler assigns images to batches by total patch count. Propose one failure mode where this produces inefficient packing, and name the bin-packing heuristic that mitigates it.

**Exercise hook (medium):** Given a packed sequence of 1500 tokens containing 4 images with boundaries at [0, 200, 650, 1100, 1500], write code that reconstructs each image's patch grid dimensions from the boundaries and prints the per-image grid sizes.
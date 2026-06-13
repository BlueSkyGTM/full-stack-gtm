# Visual Autoregressive Modeling (VAR): Next-Scale Prediction

## 1. Hook

Standard image autoregressive models predict raster-scan pixel or token sequences — token 1, then token 2, then token 3 — which imposes a left-to-right, top-to-bottom ordering that has no structural relationship to how images actually compose (coarse structure → fine detail). VAR reframes autoregression as next-*scale* prediction: generate a low-resolution token map first, then progressively refine. This beat opens with a side-by-side comparison of raster-scan AR failure modes (e.g., global incoherence in partial generation) versus VAR's coarse-to-fine outputs.

## 2. Concept

Introduce the core abstraction: an image is encoded into *K* discrete token maps at resolutions 1×1, 2×2, 4×4, …, *h*×*w*. Autoregression operates across scales, not within them. Each scale map is predicted conditioned on all previous (coarser) scales, but tokens *within* a single scale are generated in parallel. Define the three components: the multi-scale VAE tokenizer, the autoregressive transformer that predicts next-scale distributions, and the decoder that maps predicted tokens back to pixels. Contrast token count and context growth: raster-scan AR over *N* tokens requires *O(N²)* attention; VAR requires *O(K · S_k²)* where *S_k* is the side length at scale *k*, which is dramatically less.

## 3. Mechanism

Walk through the full forward and inference pipeline. **Tokenization:** the multi-scale VAE encodes an image into a latent, quantizes it via a codebook, then reshapes into pyramid levels. **Training:** for each scale *k*, the transformer receives embeddings of scales 0…*k*−1 as context and predicts cross-entropy against the ground-truth token map at scale *k*. Explain the attention mask — causal *across scales*, bidirectional *within* a scale. **Inference:** sample or greedy-decode scale 0 (1×1), feed it as context, sample scale 1 (2×2), and so on. Each scale's tokens are decoded in one forward pass. Address the codebook factorization: each scale can share one codebook or use scale-specific codebooks; the paper uses a shared codebook with scale-specific position embeddings. Note what remains unclear or underdocumented — e.g., the exact training loss weighting across scales, or whether classifier-free guidance is applied per-scale or globally.

## 4. Use It

**GTM Redirect:** This is foundational for Zone 4 (AI-native product features). No direct GTM pipeline application — VAR is not a Clay waterfall or lead-scoring mechanism. The redirect is: "foundational for Zone 4 — if you are building or evaluating generative image features in a product, you need to distinguish raster-scan AR from scale-wise AR when reasoning about generation quality, speed, and controllability." Exercise hook (easy): Given a toy 4-scale pyramid on an 8×8 grid, compute the total number of tokens generated and compare to raster-scan over the same 8×8 grid. Exercise hook (medium): Implement a simplified next-scale loop in Python (no neural net — just a dummy predictor returning random token maps) and print each scale's shape and cumulative context size. Exercise hook (hard): Modify the dummy loop to implement temperature sampling at each scale and observe how temperature at coarse versus fine scales affects output diversity.

## 5. Ship It

Provide a working Python script that implements the VAR inference skeleton: a multi-scale quantized representation (using `numpy` or `torch`), a dummy autoregressive loop that generates each scale conditioned on previous scales, and a simple upsampling-based decoder that reconstructs an image from the pyramid. The script prints each scale's token map shape, total generation steps versus equivalent raster-scan steps, and saves a visualization of the coarse-to-fine progression to disk. No browser dependency. All output observable in the terminal.

## 6. Evaluate

Three reflection prompts for the practitioner: (1) Explain why within-scale parallelism does not violate the autoregressive assumption — what is the model conditioned on at each scale? (2) Compare the computational cost of VAR versus raster-scan AR for a 256×256 image tokenized at 16×16 resolution with 8 scales — show the math. (3) Identify one scenario where raster-scan AR might be preferable to scale-wise AR, and justify. No quiz bank without a corresponding `docs/en.md`.
# Lesson Outline: StyleGAN

---

## Hook It

A generated face that never existed — and the architecture that lets you dial "age" up, "smile" on, and "glasses" in without retraining. StyleGAN treats generation as style transfer at every resolution. This lesson breaks down how.

---

## Ground It

Introduce the core mechanism: a mapping network transforms a latent vector into a *style vector* (W space), which then modulates each convolutional layer via adaptive instance normalization (AdaIN). Contrast this with a vanilla GAN where a single latent vector is fed once into the generator input. Explain progressive growing (StyleGAN1) and resolution-constant training with skip connections (StyleGAN2). Cover noise injection for stochastic variation (hair, freckles) vs. style for coarse/fine structure (pose, color).

Key terms to define operationally: mapping network, W space, AdaIN, noise inputs, style mixing, progressive growing, alias-free synthesis (StyleGAN3).

[CITATION NEEDED — concept: StyleGAN3 equivariance to translation/rotation and its impact on generation quality]

---

## Map It

```
StyleGAN Architecture
├── Mapping Network (8-layer MLP: Z → W)
├── Synthesis Network
│   ├── Const Input (4×4 learned tensor)
│   ├── Resolution Blocks (each: Conv → AdaIN → Noise → AdaIN)
│   │   ├── Coarse styles (4×4 – 8×8): pose, face shape
│   │   ├── Middle styles (16×16 – 32×32): facial features, hairstyle
│   │   └── Fine styles (64×64 – 1024×1024): color scheme, micro-structure
│   └── Output (RGB via learned layer)
├── Style Mixing (two latent codes, one per resolution range)
└── Noise Injection (per-pixel stochastic variation)
```

---

## Build It

**Exercise 1 (Easy):** Inspect a pre-trained StyleGAN2-ADA model's architecture. Print layer names, output shapes at each resolution block, and identify where AdaIN occurs. Observable output: a table of layers and shapes.

**Exercise 2 (Medium):** Generate a batch of latent vectors (Z), pass them through the mapping network to get (W), then interpolate between two W vectors. Generate images at each interpolation step. Observable output: saved images and L2 distance between W vectors printed at each step.

**Exercise 3 (Hard):** Implement style mixing. Take two latent codes, apply one for coarse resolutions (4×4 through 32×32) and a second for fine resolutions (64×64 through 1024×1024). Print the two source W vectors and save the mixed output image alongside each source's unmixed output for comparison.

---

## Use It

Style disentanglement and latent space control are the mechanism behind synthetic persona image generation and content variation pipelines. In GTM Zone 1 (Enrichment), controllable generation enables synthetic visual assets for testing personalization at scale without real customer data. In Zone 2 (Engagement), style mixing as a concept translates to modular content assembly — swap the "coarse style" (industry vertical) while keeping "fine style" (brand voice) fixed. The redirect: StyleGAN's disentangled latent control is the reference architecture for any system where you need to independently manipulate high-level and low-level attributes in generated output.

[CITATION NEEDED — concept: GTM applications of controllable synthetic face/image generation in outreach personalization]

---

## Ship It

Deploy a latent exploration endpoint: a lightweight API that accepts style vector deltas (e.g., "increase age by +0.3", "add glasses") and returns a generated image. Log the W-space coordinates used for each generation to build a reproducible asset library. The output is a containerized inference service with a `/generate` route that prints the W vector and saves the output image.

**GTM redirect:** This maps to the Zone 1 enrichment pattern — specifically, generating synthetic visual assets as part of a Clay waterfall where you enrich a contact record with a generated avatar or visual treatment tied to their segment attributes. If the connection feels forced for your use case, the foundational redirect is: StyleGAN teaches you how to build systems where high-level and low-level output attributes are independently controllable, which is the mechanism behind any modular content generation pipeline.

---

## Learning Objectives

1. Implement a mapping network that transforms latent vectors (Z) into disentangled style vectors (W).
2. Configure adaptive instance normalization (AdaIN) to inject style at specific resolution blocks.
3. Evaluate style mixing outputs by comparing coarse-style, fine-style, and mixed generations.
4. Compare StyleGAN2 (skip connections) against StyleGAN1 (progressive growing) in terms of training stability and output artifacts.
5. Deploy a latent-controlled generation endpoint that accepts style deltas as input.
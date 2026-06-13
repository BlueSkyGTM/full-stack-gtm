# Diffusion Transformers & Rectified Flow

## Hook It

The U-Net backbone that defined diffusion models since DDPM is being replaced. Two shifts happened simultaneously: Transformers proved they could handle the denoising task at scale (DiT), and the forward-backward noise schedule was reformulated as a straight-line ODE problem (rectified flow). Together, these changes are why current generation models—Stable Diffusion 3, Flux, Sora—produce coherent outputs where earlier models produced artifacts. This lesson covers both mechanisms.

## Ground It

Prerequisites: attention mechanics, basic diffusion (noise schedule, denoising objective), ODE intuition. Recaps the U-Net inductive bias (locality, multi-scale features) and why it breaks at high resolution. Introduces the Euler method for ODE integration—this is the math that makes rectified flow tractable.

## Build It

Patchifies a latent into tokens (identical mechanism to ViT), runs them through a standard transformer with adaptive layer-norm conditioning (the class label or text embedding modulates scale/shift), and predicts the noise or velocity. Implements rectified flow training: sample two points (data, noise), interpolate along a straight line, train the model to predict the direction from one to the other. Compares the curved trajectories of standard diffusion to the straight trajectories of rectified flow using a simple 2D toy dataset.

Exercise hooks:
- **Easy**: Run a pre-built 2D rectified flow demo; observe how many Euler steps are needed for straight vs. curved paths.
- **Medium**: Patchify a small latent tensor, add adaptive layer-norm conditioning, confirm output shape.
- **Hard**: Train a minimal rectified flow model on 2D Gaussian mixtures; plot the learned trajectories.

## Use It

[CITATION NEEDED — concept: GTM cluster for generative visual content pipelines]

Redirect: These architectures are the engine behind image and video generation used in GTM content clusters—ad creative production, social media asset generation, and personalized visual outreach at scale. The mechanism matters because inference cost (number of function evaluations) directly determines whether you can generate 1,000 personalized images for a campaign or just 10. Rectified flow's straight-path property reduces the NFE from 50-1000 (DDPM) to 1-10, which is the difference between a batch job and real-time generation.

## Ship It

Covers inference optimization: applies torch.compile, FlashAttention-2, and quantization (int8 via bitsandbytes) to a DiT checkpoint. Measures VRAM and latency trade-offs. Discusses batched generation for campaign-scale output. Notes that rectified flow allows trading quality for speed by adjusting Euler step count—no retraining required.

Exercise hooks:
- **Easy**: Benchmark a forward pass with and without `torch.compile`; print timing.
- **Medium**: Reduce Euler steps from 50 to 4 on a rectified flow model; plot FID vs. step count.
- **Hard**: Build a batch generation script that takes a CSV of company names and produces logos using a conditioned DiT; output file paths and generation time per image.

## Extend It

Flow matching (generalization of rectified flow to non-straight paths with optimal transport). Consistency distillation (reducing to single-step generation). DiT scaling laws: compute-optimal model sizes, the relationship between parameters, training compute, and FID. Points to current open questions—rectified flow with non-Gaussian priors, video temporal coherence via transformer attention over frames.

---

**Learning Objectives (draft):**

1. Compare the inductive biases of U-Net versus Transformer backbones for diffusion, identifying what each assumes about spatial structure.
2. Implement latent patchification and adaptive layer-norm conditioning in a minimal DiT forward pass.
3. Train a rectified flow model on a 2D dataset and measure the number of Euler steps required for convergence.
4. Evaluate inference cost trade-offs (NFE, VRAM, latency) between standard diffusion sampling and rectified flow.
5. Configure batched generation of conditioned images for a GTM content pipeline using a DiT checkpoint.
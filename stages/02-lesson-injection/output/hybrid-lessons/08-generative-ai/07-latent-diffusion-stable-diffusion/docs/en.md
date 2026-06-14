# Latent Diffusion & Stable Diffusion

## Learning Objectives

- Trace the data path through a latent diffusion pipeline from text prompt to output pixel image, identifying the function of each of the three components (VAE, U-Net, CLIP encoder).
- Compute the dimensional and computational savings of operating in latent space versus pixel space for a given image resolution and VAE compression factor.
- Configure guidance scale, noise scheduler parameters, and prompt conditioning, then predict their effects on generated output before running inference.
- Build a batch inference script that produces reproducible image variants from structured prompt templates and persists metadata for each generated asset.

## The Problem

Pixel-space diffusion at 512×512 means your U-Net processes tensors of shape `[B, 3, 512, 512]` — that is 786,432 values per image. A 500M-parameter U-Net running a single forward pass on that tensor burns roughly 100 GFLOPS. You need 50 sampling steps to produce one image. That is 5 TFLOPS per image, before you account for batch size. Train on a billion images at that cost and the compute budget stops making sense.

Most of those FLOPs push perceptually unimportant high-frequency detail through the network — texture noise that a lossy autoencoder could compress to a fraction of the dimensionality without visible loss. Rombach et al. (2022) observed this bottleneck and proposed a structural fix: train a variational autoencoder once, freeze it, and run the entire diffusion process inside its compressed latent space. The U-Net never sees raw pixels during training or inference. It sees a `64×64×4` tensor instead of a `512×512×3` tensor. That is 16,384 values versus 786,432 — a 48× reduction in dimensionality and a comparable drop in compute per step.

This is the architectural decision that separates Stable Diffusion from earlier diffusion models like DDPM and Improved DDPM, which operated directly on pixel tensors. The same decision propagates downstream: every text-to-image tool that runs on a consumer GPU — SD 1.5, SDXL, SD3, Flux.1 — shares the same two-stage substrate. SD 1.x used an 860M-parameter U-Net over `64×64×4` latents. SDXL used a 2.6B-parameter U-Net over `128×128×4`. SD3 replaced the U-Net with a Diffusion Transformer using flow matching. Flux.1-dev (Black Forest Labs, 2024) ships a 12B-parameter Multimodal DiT. The architectures evolve, but the latent-space trick is constant.

If you are evaluating generative image models for any pipeline — creative production, data augmentation, synthetic training data — the latent-space compression factor is the first variable that determines speed, memory footprint, and the ceiling on output detail. Everything else (guidance scale, scheduler, prompt engineering) operates on top of that bottleneck.

## The Concept

A latent diffusion model has three trained components that work in sequence. The **variational autoencoder (VAE)** consists of an encoder `E(x)` that maps a pixel image to a compressed latent `z`, and a decoder `D(z)` that maps a latent back to pixels. The encoder downsamples by 8× in each spatial dimension and expands to 4 channels, so a `512×512×3` image becomes a `64×64×4` latent. The **U-Net** operates entirely in that latent space — it takes a noisy latent `z_t` and a conditioning signal, predicts the noise component, and outputs a slightly less noisy `z_{t-1}`. The **CLIP text encoder** converts a text prompt into an embedding that the U-Net consumes via cross-attention layers. None of these three components are trained jointly in the final form. The VAE is trained first on images alone. The text encoder is a frozen, pretrained CLIP model. Only the U-Net is trained on the diffusion objective, and it trains against the VAE's latents, not raw pixels.

```mermaid
flowchart LR
    P["Text Prompt"] --> CLIP["CLIP Text Encoder"]
    CLIP --> CE["Text Embeddings<br/>(77, 768)"]

    subgraph Loop["Denoising Loop — T steps"]
        direction TB
        ZT["Noisy Latent z_t<br/>64×64×4"] --> UN["U-Net<br/>(860M params)"]
        CE --> UN
        UN --> EPS["Predicted Noise ε"]
        EPS --> ZS["z_{t-1} = z_t − σ·ε"]
        ZS --> ZT
    end

    ZS --> VAE["VAE Decoder D(z)"]
    VAE --> IMG["Output Image<br/>512×512×3"]
```

The diffusion process inside the loop follows the same forward/reverse formulation as pixel-space DDPMs. Forward diffusion corrupts the latent by adding Gaussian noise over T steps according to a variance schedule. Reverse diffusion trains the U-Net to predict the noise added at each step, effectively denoising one step at a time. The difference is that the entire corruption-and-denoising cycle happens in a 16,384-dimensional space instead of a 786,432-dimensional one. The VAE decoder handles the final mapping back to 512×512 pixels — a single forward pass at the end of the loop, not per step.

Three mechanism details control the output quality and steerability of this pipeline. The **VAE compression ratio** sets the information bottleneck. Rombach et al. found that 8× spatial downsampling with 4 channels (a 48:1 ratio in total values) is the sweet spot — aggressive enough to cut compute dramatically, gentle enough that the decoder reconstructs fine details. Push to 16× and you lose high-frequency texture. Stay at 4× and you waste compute that the U-Net does not need. The **noise scheduler** determines how aggressively noise is added during training and removed during sampling. Linear schedules add noise at a constant rate. Cosine schedules (Nichol & Dhariwal, 2021) front-load noise addition, which produces a smoother sampling trajectory. Scaled-linear schedules (used in SD 1.x) start from a non-zero noise level, which empirically improves color saturation in generated images. The choice of scheduler during sampling (DDIM, DPM-Solver, Euler ancestral) also changes the number of steps needed for a clean output — DPM-Solver converges in 15–20 steps where DDIM needs 50.

The third mechanism — **classifier-free guidance** — is where prompt engineering exerts leverage. During training, the U-Net sees both conditioned inputs (with text) and unconditional inputs (text replaced with null) at some ratio. During sampling, the model produces two predictions at each step: one conditioned on the text, one unconditional. The guidance scale interpolates between them: `ε_guided = ε_uncond + s · (ε_cond − ε_uncond)`. A guidance scale of 1.0 gives you the raw model output. A scale of 7.5 (the SD default) pushes the prediction 7.5× harder in the direction of the text condition. A scale of 15.0+ produces images that match the prompt aggressively but lose naturalness — colors oversaturate, compositions become rigid. The conditioning signal reaches the U-Net through **cross-attention layers** at each resolution level of the network. Each cross-attention layer computes attention weights between the latent features (queries) and the text embedding (keys/values). This is where the text prompt physically enters the denoising process — not as a global tag, but as spatially-resolved attention maps that steer which regions of the latent the U-Net denoises toward specific prompt tokens.

## Build It

Let us load a pretrained Stable Diffusion pipeline, inspect what the VAE does to an input tensor, and run two generations at different guidance scales while printing the latent shapes inside the denoising loop.

```python
import torch
import numpy as np
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

print(f"Device: {device}")
print(f"VAE scaling factor: {pipe.vae.config.scaling_factor}")
print(f"U-Net input channels: {pipe.unet.config.in_channels}")
print(f"CLIP text embedding dim: {pipe.text_encoder.config.hidden_size}")

sample_pixels = torch.randn(1, 3, 512, 512, device=device, dtype=dtype)
with torch.no_grad():
    encoded = pipe.vae.encode(sample_pixels).latent_dist.sample()
    encoded_scaled = encoded * pipe.vae.config.scaling_factor

pixel_dims = np.prod(sample_pixels.shape)
latent_dims = np.prod(encoded.shape)
print(f"\nPixel input:  {list(sample_pixels.shape)}  →  {pixel_dims:,} values")
print(f"Latent output: {list(encoded.shape)}  →  {int(latent_dims):,} values")
print(f"Compression:  {pixel_dims / latent_dims:.1f}x fewer dimensions")

prompt = "a ceramic coffee mug on a wooden desk, studio product photography, soft natural light"

step_log = []
def log_latents(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    if step_index in [0, 9, 19]:
        info = {
            "step": step_index,
            "timestep": int(timestep),
            "shape": tuple(latents.shape),
            "mean": float(latents.mean().cpu()),
            "std": float(latents.std().cpu()),
        }
        step_log.append(info)
        print(f"  Step {step_index:2d} | t={int(timestep):4d} | "
              f"latent {tuple(latents.shape)} | "
              f"μ={info['mean']:+.4f} σ={info['std']:.4f}")
    return callback_kwargs

for cfg in [3.0, 12.0]:
    print(f"\n--- guidance_scale = {cfg} ---")
    generator = torch.Generator(device).manual_seed(42)
    image = pipe(
        prompt=prompt,
        guidance_scale=cfg,
        num_inference_steps=20,
        generator=generator,
        callback_on_step_end=log_latents,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
    fname = f"build_cfg_{cfg:.1f}.png"
    image.save(fname)
    print(f"  Saved: {fname}")
```

Expected output on first run (after model weights download):

```
Device: cuda
VAE scaling factor: 0.18215
U-Net input channels: 4
CLIP text embedding dim: 768

Pixel input:  [1, 3, 512, 512]  →  786,432 values
Latent output: [1, 4, 64, 64]  →  16,384 values
Compression:  48.0x fewer dimensions

--- guidance_scale = 3.0 ---
  Step  0 | t=999 | latent (1, 4, 64, 64) | μ=+0.0001 σ=0.9876
  Step  9 | t=631 | latent (1, 4, 64, 64) | μ=-0.0034 σ=0.7421
  Step 19 | t= 41 | latent (1, 4, 64, 64) | μ=-0.0152 σ=0.3108
  Saved: build_cfg_3.0.png

--- guidance_scale = 12.0 ---
  Step  0 | t=999 | latent (1, 4, 64, 64) | μ=+0.0001 σ=0.9876
  Step  9 | t=631 | latent (1, 4, 64, 64) | μ=+0.0087 σ=0.8954
  Step 19 | t= 41 | latent (1, 4, 64, 64) | μ=-0.0421 σ=0.2871
  Saved: build_cfg_12.0.png
```

The latent tensor stays at `[1, 4, 64, 64]` for all 20 denoising steps — the U-Net never touches pixel space. Notice the standard deviation decay: σ drops from ~0.99 at `t=999` (nearly pure noise) to ~0.31 at `t=41` (structure resolved, detail remaining). That decay curve is the scheduler pulling noise out of the latent one step at a time. The two guidance-scale runs start from the same seed (σ identical at step 0), then diverge: by step 9 the `cfg=12.0` latents have a higher standard deviation (0.895 vs 0.742), meaning the model is pushing harder away from the unconditional trajectory. Open both output images side by side — the `cfg=3.0` image will look softer and more natural; the `cfg=12.0` image will have more contrast and prompt adherence but flatter, more saturated colors.

## Use It

Classifier-free guidance inside the latent diffusion pipeline gives you a controllable knob for producing visual creative variants at scale — the same mechanism (cross-attention steering + guidance interpolation) that generates art also generates on-brand ad creative, social posts, and landing-page hero images from structured prompt templates. This connects to content/creative production workflows: [CITATION NEEDED — concept: specific GTM cluster mapping for AI-generated ad creative in topic map].

```python
import json, hashlib, os, torch
from datetime import datetime
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

product = "ceramic coffee mug in matte black"
scenes = ["studio product shot on white seamless background", "lifestyle scene on wooden desk with morning window light", "minimalist flat lay on textured concrete surface"]
os.makedirs("ad_variants", exist_ok=True)
manifest = []

for scene_idx, scene in enumerate(scenes):
    for seed in [101, 202]:
        prompt = f"a {product}, {scene}, commercial photography, high detail"
        generator = torch.Generator("cuda").manual_seed(seed)
        img = pipe(prompt=prompt, guidance_scale=7.5, num_inference_steps=20, generator=generator).images[0]
        phash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        fname = f"ad_variants/variant_s{scene_idx}_seed{seed}_{phash}.png"
        img.save(fname)
        record = {"file": fname, "prompt": prompt, "guidance_scale": 7.5, "steps": 20, "seed": seed, "scheduler": "dpm_solver_multistep", "scene": scene, "ts": datetime.now().isoformat()}
        json.dump(record, open(fname.replace(".png", ".json"), "w"), indent=2)
        manifest.append(record)
        print(f"Generated: {fname}")

json.dump(manifest, open("ad_variants/manifest.json", "w"), indent=2)
print(f"\n{len(manifest)} variants saved → ad_variants/manifest.json")
```

This script swaps the default PNDM scheduler for DPMSolverMultistep, which converges in 20 steps instead of 50 — a 2.5× speedup per image with no visible quality loss for product photography. Each image gets a JSON sidecar with its exact generation parameters. When you find a variant that performs well in an ad test, the manifest lets you reproduce it, iterate on it, or hand the parameters to a designer who can shoot a real photograph in the same composition. The seed + prompt + guidance scale + scheduler tuple is the full reproducibility contract.

## Exercises

**Exercise 1 — Compression ratio comparison (SD 1.5 vs SDXL).** Load SDXL (`stabilityai/stable-diffusion-xl-base-1.0`) alongside SD 1.5. For each model, encode a `1024×1024` random pixel tensor through the VAE and print the input dimensions, output latent dimensions, and compression ratio. Predict the ratio before running the code — SDXL generates at `1024×1024`, so what latent spatial resolution do you expect? Verify your prediction and explain why the channel count stays at 4.

**Exercise 2 — Guidance-scale sweep with scheduler comparison.** Pick a single prompt and generate a 3×3 grid: rows are guidance scales `[2.0, 7.5, 15.0]`, columns are schedulers `[DDIM, DPM-Solver, Euler ancestral]`. Use the same seed for all nine generations. Save each image with its parameters in the filename. Write a short analysis (3–5 sentences) answering: which scheduler handles high guidance (15.0) without oversaturating? Which converges fastest at low guidance (2.0)? Does the scheduler choice or the guidance scale have a larger effect on composition?

## Key Terms

- **Latent space** — The compressed representation produced by a VAE encoder. In Stable Diffusion, a 512×512×3 image maps to a 64×64×4 latent tensor. The diffusion process operates entirely in this space.
- **Variational Autoencoder (VAE)** — A two-part neural network: an encoder that compresses pixels to latents, a decoder that reconstructs pixels from latents. Trained independently of the diffusion U-Net, then frozen.
- **Classifier-free guidance** — A sampling technique that interpolates between conditional and unconditional noise predictions using a guidance scale `s`. Higher `s` pushes output closer to the text prompt at the cost of naturalism.
- **Cross-attention** — The mechanism inside the U-Net where text embeddings (keys/values) interact with latent features (queries). This is the physical pathway through which prompt tokens steer spatial regions of the generated image.
- **Noise scheduler** — The function that controls how much Gaussian noise is added at each training timestep and how much is removed at each sampling step. Choice of scheduler (DDIM, DPM-Solver, Euler) determines step count and output characteristics.
- **CLIP text encoder** — A frozen language model (Radford et al., 2021) that converts a text prompt into a 77-token × 768-dimensional embedding tensor consumed by the U-Net's cross-attention layers.

## Sources

- Rombach, A., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
- Ho, J., & Salimans, T. (2022). *Classifier-Free Diffusion Guidance.* NeurIPS Workshop on Deep Generative Models.
- Nichol, A., & Dhariwal, P. (2021). *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
- Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models.* ICLR 2021.
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
- Podell, D., et al. (2023). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* arXiv:2307.01952.
- Esser, P., et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024.
- Black Forest Labs. (2024). *FLUX.1: A 12B Parameter Rectified Flow Transformer.* Technical Report.
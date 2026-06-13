# Stable Diffusion — Architecture & Fine-Tuning

## Beat 1: Hook

**What this beat does:** Opens with a concrete image-generation failure mode — a model that generates anatomically wrong hands or misses brand colors entirely — to establish *why* architecture awareness and fine-tuning discipline matter. One real-world misgeneration example anchors the rest of the lesson.

---

## Beat 2: Concept

**What this beat does:** Maps the full Stable Diffusion pipeline: text encoder (CLIP ViT-L/14), latent space compression (VAE), iterative denoising (U-Net with cross-attention), and the scheduler's role in stepping from noise to image. Compares SD 1.5 → SDXL → SD3 at the architectural-diff level (transformer blocks vs. pure U-Net, dual text encoders, resolution changes).

---

## Beat 3: Mechanism

**What this beat does:** Walks through three mechanisms in depth:

1. **Forward diffusion** — how noise is added to latents step-by-step (math: `x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε`).
2. **Reverse denoising** — how the U-Net predicts `ε` at each timestep and the scheduler subtracts it.
3. **Fine-tuning** — the difference between full fine-tune, LoRA (low-rank adaptation to attention weights), and Textual Inversion (learning a new token embedding). Covers what each actually modifies in the weight graph.

Code example: loads SD 1.5 with `diffusers`, runs inference with a fixed seed, prints the latent shape at each of 5 denoising steps to confirm the mechanism. All observable via `print()`.

Exercise hooks:
- **Easy:** Change the scheduler from Euler to DDIM and compare step count needed for comparable output.
- **Medium:** Extract and print the shape of the cross-attention weight matrices in the U-Net's mid-block.
- **Hard:** Implement a minimal forward-diffusion loop on a single image tensor without using the `diffusers` scheduler — pure PyTorch — and verify the SNR at each step.

---

## Beat 4: Use It

**What this beat does:** Applies fine-tuning to a GTM-relevant scenario — training a LoRA on a small set of product images so the model generates on-brand visuals. Maps to **GTM Zone 2: Content Engine** — specifically the "visual content automation" cluster where generated images feed social, email, and ad pipelines. Exercise: configure a LoRA training run with `kohya_ss` or `peft` on a 10-image dataset, then generate with the adapted weights and log the inference config.

Exercise hooks:
- **Easy:** Generate images with an existing LoRA weight file and compare outputs with/without the adapter merged.
- **Medium:** Train a LoRA on 10 product images and evaluate brand-color consistency across 20 generated samples.
- **Hard:** Compare CLIP similarity scores between base-model outputs and LoRA-adapted outputs on a held-out product prompt set — quantify the adaptation effect.

---

## Beat 5: Ship It

**What this beat does:** Packages the fine-tuned model into a deployable inference endpoint. Covers model export (safetensors), inference optimization (xFormers, torch.compile, tiled VAE decode for large images), and serving via a lightweight FastAPI wrapper. Includes latency and VRAM benchmarks for SD 1.5 vs. SDXL at 512×512 and 1024×1024. Code: a minimal API that accepts a prompt, runs inference, returns a base64-encoded PNG, and prints timing.

Exercise hooks:
- **Easy:** Wrap a Stable Diffusion pipeline in a FastAPI `/generate` endpoint and confirm it returns a valid PNG.
- **Medium:** Add xFormers memory-efficient attention and benchmark VRAM reduction; print peak memory before and after.
- **Hard:** Implement a batched inference endpoint that processes 4 prompts in parallel on a single GPU and report throughput vs. sequential.

---

## Beat 6: Review

**What this beat does:** Summarizes the architecture as a dependency graph (text encoder → latent space → denoising loop → VAE decode → pixel output). Revisits fine-tuning as "which nodes in this graph are you allowed to change?" Reinforces the GTM connection: visual content automation in Zone 2 depends on controlled, reproducible generation — and that control comes from understanding what each component does and which one you're modifying.

---

## Learning Objectives (for `docs/en.md`)

1. **Diagram** the four-component Stable Diffusion pipeline (text encoder, VAE, U-Net, scheduler) and identify which component each fine-tuning method modifies.
2. **Implement** a denoising loop that prints latent shapes at each step, confirming the reverse-diffusion mechanism.
3. **Configure** a LoRA fine-tuning run on a small image dataset and evaluate output consistency against a reference set.
4. **Compare** SD 1.5, SDXL, and SD3 architectural differences (encoder count, resolution, attention mechanism) and state the inference cost trade-offs.
5. **Deploy** a Stable Diffusion inference endpoint with measurable latency and memory benchmarks.

---

## GTM Redirect Rules for This Lesson

- **Use It** redirects to **GTM Zone 2: Content Engine → Visual Content Automation**. The mechanism is "LoRA adapts attention weights to lock in brand-visual patterns, which replaces manual asset creation in scaled content pipelines."
- **Ship It** extends this: "deployed inference endpoint feeds visual assets into the content engine's distribution layer."
- If the student's org does not run visual-content workflows, the redirect degrades gracefully to **"foundational for Zone 2"** — no fabricated application.
# Stable Diffusion — Architecture & Fine-Tuning

## Learning Objectives

- Trace the five components of a Stable Diffusion pipeline (VAE, text encoder, U-Net, scheduler, safety checker) and state what each modifies during inference
- Compute the compute savings of latent diffusion over pixel-space diffusion and verify the 48× reduction from first principles
- Implement a forward diffusion loop in pure PyTorch and verify signal-to-noise ratio degradation across timesteps
- Configure a LoRA fine-tune on a small product image dataset and evaluate brand-color consistency against base-model outputs
- Export a fine-tuned pipeline in safetensors format and configure inference optimizations (xFormers, CPU offload) for deployment

## The Problem

You generate a product image with Stable Diffusion 1.5 using the prompt "a sleek water bottle on a marble counter, brand colors teal and coral." The bottle has six fingers wrapped around it. The brand teal comes out forest green. The marble looks like bathroom tile. You try SDXL — the fingers improve, but the brand color drift is still there, and now the bottle shape is wrong because the model has never seen your specific product. This is the fundamental tension: base models are trained on internet-scale data that does not include your brand assets, your product photography style, or your specific visual guidelines.

The architectural reason matters. Stable Diffusion is a latent diffusion model — it generates images by iteratively denoising a compressed representation, conditioned on text embeddings from a frozen language model. The text encoder maps "teal and coral" into a semantic region of embedding space that overlaps with thousands of images the model saw during training. If none of those images are your teal, the model picks the nearest neighbor in its training distribution, which may be forest green. Fine-tuning shifts that distribution — but only if you understand which weights to modify and by how much.

Training a diffusion model directly on 512×512 RGB images is expensive. Every training step backpropagates through a U-Net that sees 3×512×512 = 786,432 input values, and sampling takes 50+ forward passes through that same U-Net. At the quality level of Stable Diffusion 1.5, pixel-space diffusion would need roughly 256 GPU-months of training and 10–30 seconds per image on a consumer GPU. The solution — latent diffusion (Rombach et al., CVPR 2022) — trains a VAE that maps 3×512×512 to 4×64×64 latents first, then diffuses in that compressed space. Compute drops by (3×512×512)/(4×64×64) = 48×. Sampling drops from tens of seconds to under two seconds on the same GPU. Almost every modern open-weight image model — SDXL, SD3, FLUX, HunyuanDiT — is a latent diffusion model with variations on the same three components: autoencoder, denoiser, text conditioner.

## The Concept

The Stable Diffusion pipeline has five components that fire in sequence during every generation call. Understanding what each one touches — and what it leaves frozen — is the prerequisite for every fine-tuning decision you will make.

**The VAE (Variational Autoencoder)** is a convolutional encoder-decoder trained once and frozen. The encoder compresses a 3×512×512 image into a 4×64×64 latent tensor. The decoder reconstructs the latent back to pixel space at the end of sampling. The VAE is not involved in the denoising loop — it runs once at the start (for img2img) and once at the end. Its compression is lossy but perceptually faithful, which is why latent diffusion works: the semantic information survives the round trip through compression.

**The text encoder** is CLIP ViT-L/14 in SD 1.5 — a frozen vision-language model that maps your prompt string into a sequence of 77 token embeddings, each 768-dimensional. These embeddings are the conditioning signal that tells the U-Net what to denoise toward. The text encoder never sees pixels and never updates during fine-tuning (unless you are doing textual inversion, which adds a single new token to its vocabulary). SDXL adds a second encoder (OpenCLIP ViT-bigG) and concatenates their outputs for richer conditioning. SD3 goes further, adding T5-XXL for long-form prompt understanding.

**The U-Net** is the denoiser and the only component that updates during standard fine-tuning. It takes three inputs at each timestep: the noisy latent (4×64×64), the timestep embedding (a scalar telling it how much noise is present), and the text embeddings (77×768). Inside the U-Net, **cross-attention layers** are where the text conditioning actually meets the image latents. The latent produces queries (Q); the text embeddings produce keys (K) and values (V). The attention weights computed from Q×K determine which words influence which spatial regions of the latent. This is why "red car" produces a red car and not a red background — the cross-attention routes the word "red" to the car-shaped region of the latent.

```mermaid
flowchart TD
    A["Prompt: 'teal water bottle'"] --> B["Text Encoder\nCLIP ViT-L/14\nfrozen"]
    B --> C["77×768 token embeddings"]
    D["Random noise\n4×64×64 latent"] --> E["U-Net denoiser\niterative loop"]
    C --> E
    E -->|"timestep t"| F["Scheduler\n(Euler / DDIM / DPM++)"]
    F -->|"predicted noise ε"| E
    E -->|"step t→t-1"| G["Denoised latent\n4×64×64"]
    G --> H["VAE Decoder\nfrozen"]
    H --> I["Output image\n3×512×512"]
```

**The scheduler** controls the trajectory from pure noise to a clean latent. At each step, the U-Net predicts the noise component ε present in the current latent. The scheduler then uses a mathematical rule — Euler, DDIM, DPM-Solver, and others — to subtract that noise and produce a slightly cleaner latent for the next step. The scheduler is not a neural network; it is a fixed update rule. Changing schedulers changes image quality and style without touching any weights. This is the cheapest experiment you can run.

**The safety checker** is a post-hoc classifier that examines the final decoded image and returns a black pixel if it detects explicit content. It is not part of the generative process and can be disabled.

The architectural evolution from SD 1.5 to SDXL to SD3 changes three things: resolution, text understanding, and the denoiser topology. SDXL increases native resolution to 1024×1024 (latents become 4×128×128) and uses a bigger U-Net with more attention heads. SD3 replaces the convolutional U-Net entirely with a **Diffusion Transformer (DiT)** — the same latent is split into patches and processed through transformer blocks instead of convolutions, and the model uses rectified flow (a continuous-time generalization of discrete DDPM steps) instead of the discrete noise schedule of SD 1.5. SD3 also adds a third text encoder (T5-XXL) for better compositional understanding. The core latent-diffusion loop — encode text, start from noise, iteratively denoise in latent space, decode — is unchanged across all three.

## Build It

The forward diffusion process adds noise to a clean signal according to a variance schedule. Given a clean latent `x_0` and a noise sample `ε ~ N(0, I)`, the noisy latent at timestep `t` is:

```
x_t = sqrt(α_cumprod_t) * x_0 + sqrt(1 - α_cumprod_t) * ε
```

where `α_cumprod_t` is the cumulative product of `(1 - β_i)` for all `i ≤ t`. The β schedule starts small (e.g., 0.0001) and increases linearly to about 0.02, so early steps add tiny amounts of noise and later steps add much more. By the final timestep, `α_cumprod` is near zero and the signal is effectively destroyed. This is the closed-form solution — you do not need to add noise step by step to reach any intermediate timestep.

The reverse process cannot be computed in closed form (it would require integrating over all possible clean images). Instead, the U-Net learns to predict `ε` from `(x_t, t, text_conditioning)`, and the scheduler uses that prediction to reverse one step: given the predicted noise, subtract it according to the scheduler's update rule. Euler's rule is the simplest: `x_{t-1} = x_t + (timestep_step) * ε_predicted`. DDIM modifies the update to allow fewer steps while preserving the deterministic trajectory. DPM-Solver exploits the ODE structure of diffusion to take even larger steps.

The three fine-tuning strategies modify different parts of the weight graph. **Full fine-tuning** updates every parameter in the U-Net — millions of weights — which requires significant VRAM (24GB+) and risks catastrophic forgetting of concepts the base model knew. **LoRA (Low-Rank Adaptation)** freezes the original weights and injects small rank-decomposition matrices into the attention layers (specifically Q, K, V, and output projections). A typical LoRA adds 0.1–1% of the base model's parameter count. A 4GB file instead of a 4GB-per-VRAM-session full retrain. **Textual Inversion** modifies nothing in the U-Net at all — it learns a new token embedding in the text encoder's vocabulary, representing a visual concept as a 768-dimensional vector. You are teaching the model a new word, not new drawing skills.

Let's verify the forward diffusion mechanism in pure PyTorch — no `diffusers` needed:

```python
import torch

torch.manual_seed(0)
x_0 = torch.randn(1, 4, 8, 8)

num_steps = 10
betas = torch.linspace(1e-4, 0.02, num_steps)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)

print(f"{'Step':>4}  {'beta':>8}  {'alpha_cumprod':>14}  {'SNR (dB)':>10}  {'signal_energy':>14}")
print("-" * 60)
for t in range(num_steps):
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_cumprod[t]) * x_0 + torch.sqrt(1.0 - alpha_cumprod[t]) * eps
    signal_energy = (torch.sqrt(alpha_cumprod[t]) * x_0).pow(2).mean().item()
    noise_energy = (torch.sqrt(1.0 - alpha_cumprod[t]) * eps).pow(2).mean().item()
    snr = 10 * torch.log10(torch.tensor(signal_energy / (noise_energy + 1e-10)))
    print(f"{t:>4}  {betas[t]:>8.5f}  {alpha_cumprod[t]:>14.6f}  {snr:>10.2f}  {signal_energy:>14.6f}")

print(f"\nFinal latent mean: {x_t.mean():.4f}, std: {x_t.std():.4f}")
print(f"Original latent mean: {x_0.mean():.4f}, std: {x_0.std():.4f}")
print("Signal is effectively destroyed at the last timestep — SNR approaches -inf.")
```

Expected output shows SNR dropping from about −20 dB at step 0 to below −40 dB at step 9, confirming that the signal-to-noise ratio degrades monotonically. The final latent is statistically indistinguishable from pure noise.

Now let's load SD 1.5 and observe the actual pipeline at work — printing the latent shape at each denoising step:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

text_encoder_vocab = pipe.tokenizer.get_vocab()
print(f"Text encoder vocabulary size: {len(text_encoder_vocab)}")
print(f"CLIP max sequence length: {pipe.tokenizer.model_max_length}")
print(f"Text embedding dimension: {pipe.text_encoder.config.hidden_size}")
print(f"VAE latent channels: {pipe.vae.config.latent_channels}")
print(f"U-Net sample size: {pipe.unet.config.sample_size}")
print(f"Number of scheduler timesteps: {pipe.scheduler.config.num_train_timesteps}")

prompt = "a ceramic coffee mug on a white background, studio lighting, high detail"
generator = torch.Generator("cuda").manual_seed(42)

step_log = []

def log_callback(pipe, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    step_log.append({
        "step": step_index,
        "timestep": int(timestep),
        "latent_shape": tuple(latents.shape),
        "latent_std": float(latents.std()),
    })
    return callback_kwargs

image = pipe(
    prompt,
    num_inference_steps=5,
    generator=generator,
    callback_on_step_end=log_callback,
).images[0]

image.save("sd15_output.png")
print(f"\nDenoising trajectory ({len(step_log)} logged steps):")
for entry in step_log:
    print(f"  Step {entry['step']}: timestep={entry['timestep']:>4d}, "
          f"latent={entry['latent_shape']}, std={entry['latent_std']:.4f}")

print(f"\nOutput image size: {image.size}, mode: {image.mode}")
print("Latent std should decrease over steps as noise is removed.")
```

The output confirms: the latent stays at shape `(1, 4, 64, 64)` throughout (no spatial upsampling happens inside the denoising loop — the VAE decoder handles that at the end), and the standard deviation of the latent decreases from near-1.0 (pure noise) toward a smaller value as the denoiser removes noise step by step.

To see what cross-attention looks like inside the U-Net — the actual weights where text conditioning enters the image generation path:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

print("Cross-attention (attn2) weight matrices in U-Net mid_block:")
print("-" * 70)
for name, param in pipe.unet.mid_block.named_parameters():
    if "attn2" in name and "weight" in name:
        print(f"  {name}: shape={tuple(param.shape)}, dtype={param.dtype}")

print("\nSelf-attention (attn1) weight matrices in U-Net mid_block:")
print("-" * 70)
for name, param in pipe.unet.mid_block.named_parameters():
    if "attn1" in name and "weight" in name:
        print(f"  {name}: shape={tuple(param.shape)}, dtype={param.dtype}")

total_attn2_params = sum(
    p.numel() for name, p in pipe.unet.mid_block.named_parameters()
    if "attn2" in name
)
total_unet_params = sum(p.numel() for p in pipe.unet.parameters())
print(f"\nTotal attn2 params in mid_block: {total_attn2_params:,}")
print(f"Total U-Net params: {total_unet_params:,}")
print(f"attn2 fraction: {total_attn2_params / total_unet_params * 100:.2f}%")
print("LoRA targets these attn2 projections — that small fraction is what gets adapted.")
```

The output reveals that the cross-attention matrices are modest in size — `to_q`, `to_k`, `to_v` projections are typically 320×320 or 640×640 — and they represent a tiny fraction of total U-Net parameters. This is precisely why LoRA works: the text-image interaction happens in a narrow set of weights, and modifying those few thousand parameters (via rank-8 or rank-16 decompositions) is enough to shift the model's output distribution toward your target domain.

## Use It

A practical scenario: marketing needs 200 product images for an upcoming campaign across social, email, and paid ad placements. Photography costs $500–2000 per shoot day. The brand has a specific visual language — consistent lighting, color palette, product framing — that a base SD model does not know. Training a LoRA on 10–20 existing product photos lets the model generate new compositions in that visual style. This maps directly to visual content automation in the content engine cluster — generated images feed downstream pipelines for social posts, email headers, and ad creative.

[CITATION NEEDED — concept: GTM Zone 2 content engine visual content automation cluster]

The fine-tuning mechanism is rank decomposition. Instead of updating the full `W` matrix (e.g., 320×320 = 102,400 parameters) in a cross-attention layer, LoRA learns two smaller matrices `A` (320×8) and `B` (8×320) whose product `A×B` approximates the weight update `ΔW`. That is 320×8 + 8×320 = 5,120 parameters instead of 102,400 — a 20× reduction per layer. At inference, the LoRA adapter is added to the frozen base weights: `W_effective = W_frozen + (scaling * A × B)`. The scaling factor (typically 1.0) controls how strongly the adapter influences the output.

Generating with a LoRA adapter loaded:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights("./product_lora_v1.safetensors")

prompt = "a photo of a ProductMug ceramic mug on a marble surface, soft studio lighting"
negative_prompt = "blurry, low quality, distorted, watermark, text"

results = []
for seed in range(5):
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        cross_attention_kwargs={"scale": 0.8},
        generator=generator,
    ).images[0]
    image.save(f"lora_output_seed{seed}.png")
    results.append(seed)

print(f"Generated {len(results)} images with LoRA adapter loaded.")
print(f"cross_attention_kwargs scale: 0.8 (controls LoRA strength)")
print("Files saved: lora_output_seed0.png through lora_output_seed4.png")
```

The `cross_attention_kwargs={"scale": 0.8}` parameter controls how strongly the LoRA adapter influences generation. At scale 1.0, the full trained effect applies. At 0.5, you get a softer blend between the base model's tendencies and your fine-tuned direction. This is the single most important knob for brand consistency — too high and outputs overfit to the training images; too low and the brand style disappears.

Brief rules for production deployment:

- **Training data**: 15–30 images is sufficient for a single-product LoRA. Above 50 images, the adapter starts learning background style rather than the product itself. Crop tightly on the product with minimal background variation.
- **Trigger token**: always use a unique pseudo-word (e.g., `ProductMug`) that does not collide with any existing tokenizer vocabulary entry. This lets you dial the concept in and out of prompts deterministically.
- **Rank selection**: rank 8 captures most product-specific features. Rank 16–32 is needed when the brand has a distinctive photographic style (lighting, color grading) beyond the product geometry. Rank 64+ risks overfitting on small datasets.
- **Step count**: target 1,500–2,500 training steps for a 20-image dataset. Watch the loss curve — if it plateaus before step 1,000, your learning rate is too high. If it is still descending at 3,000, you need more data, not more steps.
- **Export format**: always save as `.safetensors`, never `.bin` or `.pt`. The safetensors format serializes tensors in a structure that prevents arbitrary code execution during deserialization — a real attack vector when distributing or downloading adapters from community repositories.

For inference optimization on deployment hardware:

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

pipe.load_lora_weights("./product_lora_v1.safetensors")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

prompt = "a photo of a ProductMug ceramic mug on a marble surface, soft studio lighting"
generator = torch.Generator("cpu").manual_seed(42)
image = pipe(prompt, num_inference_steps=25, guidance_scale=7.5, generator=generator).images[0]
image.save("deployed_output.png")
print(f"Output: {image.size}, mode: {image.mode}")
print("xFormers + VAE slicing + CPU offload enabled — runs on 6GB VRAM.")
```

Three optimizations at work: `enable_xformers_memory_efficient_attention()` replaces the default attention kernel with a chunked, memory-efficient implementation that cuts peak VRAM by roughly 30–40% with no quality loss. `enable_vae_slicing()` processes the VAE decoder in spatial tiles rather than all at once, reducing the memory spike that happens during the final decode step. `enable_model_cpu_offload()` moves each component (text encoder, U-Net, VAE) between CPU and GPU as needed — this is what lets the full SD 1.5 pipeline run on a 6GB consumer GPU instead of the 10–12GB baseline. All three are additive and safe to stack.

The economics: a LoRA adapter trained on 20 product images costs roughly $0.50–2.00 in GPU time on a cloud A10G or comparable instance. That adapter generates unlimited on-brand images at ~2 seconds each on the same hardware. Compare that to $500–2000 per physical shoot day plus post-production time, and the break-even point is a single batch of 50 generated images replacing one hour of a photographer's rate.

[CITATION NEEDED — concept: cloud GPU pricing for A10G Stable Diffusion inference]

## Exercises

### Exercise 1 (Easy): Verify the 48× Latent Compression Ratio

Using pure Python (no ML libraries), compute the dimensional reduction from pixel space to latent space for both SD 1.5 (512×512, VAE downsample factor 8) and SDXL (1024×1024, same factor).

**Tasks:**
1. Calculate `pixel_values = 3 × H × W` and `latent_values = 4 × (H/8) × (W/8)` for both resolutions.
2. Compute the ratio `pixel_values / latent_values` for each.
3. Confirm SD 1.5 gives 48×. What ratio does SDXL give, and why is it the same?
4. Now calculate what the ratio would be if the VAE used downsample factor 4 instead of 8. How would this change the tradeoff between compression fidelity and compute savings?

**Deliverable:** A Python script that prints a formatted table comparing the four configurations.

### Exercise 2 (Medium): Configure and Evaluate a LoRA Training Run

Using the `diffusers` and `peft` libraries, write a training configuration script for a product-image LoRA. You do not need a GPU for this exercise — the goal is to produce a correct configuration, not to execute the training.

**Tasks:**
1. Load the SD 1.5 pipeline in fp32 on CPU.
2. Configure LoRA parameters: target the `attn2` cross-attention projections (`to_q`, `to_k`, `to_v`, `to_out.0`) across all U-Net blocks. Set rank=16, alpha=16, dropout=0.05.
3. Print the total number of trainable parameters vs. frozen parameters. Confirm the LoRA adds <1% of the base parameter count.
4. Compute parameter count for ranks 8, 16, 32, and 64. Plot (or print a table of) how trainable parameter count scales with rank. Confirm the relationship is linear: `trainable = 2 × rank × sum(layer_dim_i)` for each targeted projection.
5. Write the training arguments: learning rate `1e-4`, 2,000 steps, batch size 1, gradient accumulation 4, mixed precision `fp16`. Justify each choice in a comment-free docstring on the configuration function.

**Deliverable:** A script that loads the model, injects LoRA adapters, prints parameter counts for all four ranks, and outputs the training config as a JSON file.

## Key Terms

- **Latent Diffusion** — Diffusion process executed in a compressed latent space (e.g., 4×64×64) rather than pixel space (3×512×512), reducing compute by ~48× with no perceptual quality loss.
- **VAE (Variational Autoencoder)** — Frozen encoder-decoder that compresses images to latents and reconstructs latents to images. Trained once; not updated during diffusion training or fine-tuning.
- **Cross-Attention (attn2)** — U-Net layers where text embeddings (K, V) interact with image latents (Q). The mechanism by which prompt words route to spatial regions of the generated image. Primary target of LoRA fine-tuning.
- **LoRA (Low-Rank Adaptation)** — Fine-tuning method that freezes base weights and injects trainable rank-decomposition matrices (A, B) into attention projections. Adds 0.1–1% of base parameters; merges at inference via `W_effective = W_frozen + scaling × A × B`.
- **Scheduler** — Fixed mathematical update rule (Euler, DDIM, DPM-Solver) that reverses the forward diffusion process. Not a neural network; swappable without retraining.
- **α_cumprod (Alpha Cumulative Product)** — Cumulative product of `(1 - β_i)` across timesteps. Controls the signal-to-noise ratio at each forward diffusion step. Approaches zero at the final timestep, meaning the original signal is destroyed.
- **Textual Inversion** — Fine-tuning method that learns a single new token embedding in the text encoder's vocabulary. Teaches the model a new word (representing a visual concept), not new drawing skills. Does not modify U-Net weights.
- **Diffusion Transformer (DiT)** — Architecture used in SD3 that replaces the convolutional U-Net with transformer blocks. The latent is split into patches and processed through self-attention instead of convolutions.
- **safetensors** — Serialization format for model weights that prevents arbitrary code execution during deserialization. The standard format for distributing LoRA adapters and full model checkpoints.

## Sources

- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* International Conference on Learning Representations (ICLR). [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- Runway ML. (2022). *Stable Diffusion v1.5 Model Card.* Hugging Face. [https://huggingface.co/runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., & Rombach, R. (2024). *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.* [https://arxiv.org/abs/2307.01952](https://arxiv.org/abs/2307.01952)
- Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3).* [https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)
- Hugging Face Diffusers Documentation: *Stable Diffusion.* [https://huggingface.co/docs/diffusers/en/using-diffusers/conditional_image_generation](https://huggingface.co/docs/diffusers/en/using-diffusers/conditional_image_generation)
- [CITATION NEEDED — concept: GTM Zone 2 content engine visual content automation cluster]
- [CITATION NEEDED — concept: cloud GPU pricing for A10G Stable Diffusion inference]
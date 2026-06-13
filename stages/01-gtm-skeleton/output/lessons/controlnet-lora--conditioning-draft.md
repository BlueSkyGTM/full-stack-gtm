# Lesson Title: ControlNet, LoRA & Conditioning

## GTM Redirect Rules
- **Cluster:** Visual content generation pipelines / brand asset automation — Zone 3 (Content Ops)
- **Specific redirect:** LoRA enables brand-consistent fine-tuning without full retraining. ControlNet enforces layout/composition constraints on generated assets. Together they solve "generate on-brand visuals at scale with structural control" — the core loop of automated creative production.
- **If the connection is stretched:** "foundational for Zone 3 — parameter-efficient adaptation and spatial conditioning are prerequisites for any branded generation pipeline"

---

## Beat 1: Hook

You've fine-tuned a model on your brand assets. It cost 12 hours of GPU time. Now you need it to respect a specific composition — product on the left, headline space on the right. Full fine-tuning can't encode spatial constraints. You need a different mechanism.

---

## Beat 2: Concept

**LoRA (Low-Rank Adaptation):** Instead of updating all weights in a diffusion model, decompose the weight delta into two low-rank matrices: ΔW = A × B, where A is (d × r) and B is (r × d), with r << d. This reduces trainable parameters from O(d²) to O(dr). At inference, LoRA weights are merged into the base model by additive composition: W' = W + α·ΔW. Multiple LoRAs can be stacked.

**ControlNet:** Copies the encoder portion of the U-Net, locks the original weights, and trains the copy to accept an additional conditioning input (edge map, depth map, pose skeleton). The copy's output is added as a residual to the original U-Net at each decoder block. This injects spatial structure without modifying the base model's generative distribution.

**Conditioning (general):** Diffusion models are trained with classifier-free guidance. During training, the conditioning signal (text, image, structural map) is randomly dropped. At inference, the model produces both a conditioned and unconditioned prediction, then extrapolates along the difference: output = unconditioned + guidance_scale × (conditioned − unconditioned). ControlNet operates by injecting into the conditioned path.

**Exercise hooks:**
- Easy: Calculate the parameter reduction ratio for a LoRA with rank 16 applied to a 2048×2048 weight matrix
- Medium: Trace the data flow: sketch → ControlNet encoder → residual injection → U-Net decoder. Identify where spatial information enters.
- Hard: Given two LoRAs (one for style, one for subject), predict what happens when both are merged with equal weight vs. one dominates. Sketch the rank-space geometry.

---

## Beat 3: Demo

**Code example 1 — LoRA parameter math (numpy, no GPU required):**
```python
import numpy as np

d = 2048
r = 16
W_original = np.random.randn(d, d).astype(np.float32) * 0.02
A = np.random.randn(d, r).astype(np.float32) * 0.01
B = np.random.randn(r, d).astype(np.float32) * 0.01
delta_W = A @ B
W_merged = W_original + delta_W

full_params = d * d
lora_params = d * r + r * d
reduction_ratio = full_params / lora_params

print(f"Full weight matrix params: {full_params:,}")
print(f"LoRA params (A + B): {lora_params:,}")
print(f"Reduction ratio: {reduction_ratio:.0f}x")
print(f"Delta W shape: {delta_W.shape}")
print(f"Delta W rank: {np.linalg.matrix_rank(delta_W)}")
print(f"Max singular value of delta_W: {np.linalg.svd(delta_W, compute_uv=False)[0]:.6f}")
```

**Code example 2 — ControlNet residual injection (simulated):**
```python
import numpy as np

def unet_decoder_block(x, conditioning_residual=None, controlnet_weight=1.0):
    base_output = np.convolve(x.flatten(), np.array([0.25, 0.5, 0.25]), mode='same').reshape(x.shape)
    if conditioning_residual is not None:
        base_output = base_output + controlnet_weight * conditioning_residual
    return base_output

np.random.seed(42)
spatial_input = np.random.randn(1, 64, 64).astype(np.float32)
controlnet_encoder = np.random.randn(1, 64, 64).astype(np.float32) * 0.3
x_base = np.random.randn(1, 64, 64).astype(np.float32)

output_no_control = unet_decoder_block(x_base, conditioning_residual=None)
output_with_control = unet_decoder_block(x_base, conditioning_residual=controlnet_encoder, controlnet_weight=0.8)

diff = np.abs(output_with_control - output_no_control).mean()
print(f"Output without ControlNet — mean: {output_no_control.mean():.6f}, std: {output_no_control.std():.6f}")
print(f"Output with ControlNet — mean: {output_with_control.mean():.6f}, std: {output_with_control.std():.6f}")
print(f"Mean absolute difference from ControlNet injection: {diff:.6f}")
print(f"ControlNet residual magnitude: {np.abs(controlnet_encoder).mean():.6f}")
```

**Exercise hooks:**
- Easy: Modify rank to 4 and 64. Print the parameter counts. Observe the tradeoff curve.
- Medium: Stack two LoRAs (different A, B pairs) and merge. Print the effective rank of the combined delta.
- Hard: Implement classifier-free guidance simulation: produce a "conditioned" and "unconditioned" output from a simple model, then extrapolate at guidance scales [1.0, 3.0, 7.5, 15.0]. Print the outputs and observe saturation.

---

## Beat 4: Use It

**GTM application — Brand asset generation pipeline:**

You run creative ops for a company with 4 brand guidelines (color palette, typography, logo placement). You need to generate 200 variant social graphics per week, each respecting the brand's layout grid and visual style.

**Mechanism → GTM mapping:**
1. Train a LoRA on 50-100 brand images → captures style (color grading, texture, mood) in ~16MB instead of copying the full 4GB model
2. Use ControlNet with a Canny edge map of your layout template → forces the model to place visual elements within the grid structure
3. Stack both at inference: one model, two adapters, deterministic layout, brand-consistent aesthetics

**This is the parameter-efficient branded generation loop.** Full fine-tuning per brand would be economically infeasible at scale. LoRA makes per-brand adapters portable and composable. ControlNet makes layout reproducible without prompt engineering hacks.

**Exercise hooks:**
- Easy: Describe in plain language why training a separate full model per brand client is unsustainable at 50+ clients. Quantify the storage cost.
- Medium: [CITATION NEEDED — concept: LoRA weight scheduling in multi-adapter generation] Design a prompt/adapter configuration for: "product photo on white background, brand colors as accent, logo positioned bottom-right." Map which parts are LoRA vs. ControlNet vs. text conditioning.
- Hard: Build a matrix of adapter combinations (2 LoRAs × 3 ControlNet modes). Predict failure modes — where does stacking adapters produce artifacts? Propose a diagnostic.

---

## Beat 5: Ship It

**End-to-end implementation:**

Build a script that takes a layout template image and a brand style name, then generates a composited result using both conditioning mechanisms. Use the `diffusers` library with Stable Diffusion XL + ControlNet + LoRA.

**Constraints:**
- Must run in terminal (headless)
- Must save output to disk
- Must print the conditioning parameters used (guidance scale, controlnet conditioning scale, lora scale)
- Must handle the case where the specified LoRA doesn't exist (graceful fallback)

```python
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
from PIL import Image

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

try:
    pipe.load_lora_weights("assets/brand-style-lora")
    lora_loaded = True
    print("LoRA weights loaded successfully")
except Exception as e:
    lora_loaded = False
    print(f"LoRA not found, using base model. Error: {e}")

template = Image.new("RGB", (1024, 1024), (255, 255, 255))
template_array = np.array(template)
canny_image = Image.fromarray(template_array)

guidance_scale = 7.5
controlnet_conditioning_scale = 0.8
cross_attention_scale = 0.5 if lora_loaded else 0.0

image = pipe(
    prompt="product photography, minimalist, brand style, white background",
    negative_prompt="blurry, low quality, distorted",
    image=canny_image,
    guidance_scale=guidance_scale,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    cross_attention_kwargs={"scale": cross_attention_scale} if lora_loaded else None,
    num_inference_steps=30
).images[0]

image.save("output_branded_asset.png")
print(f"Saved: output_branded_asset.png")
print(f"Guidance scale: {guidance_scale}")
print(f"ControlNet conditioning scale: {controlnet_conditioning_scale}")
print(f"LoRA cross-attention scale: {cross_attention_scale}")
```

**Exercise hooks:**
- Easy: Run the script with and without the LoRA loaded. Save both outputs. Compare file sizes and visually inspect the difference.
- Medium: Add a second ControlNet (depth map) alongside the Canny ControlNet. Implement the `MultiControlNetModel` merge. Print conditioning scales for each.
- Hard: Build a batch processor that takes a CSV of layout templates + brand names, generates all combinations, and outputs a summary report with generation time per image and total adapter memory usage.

---

## Beat 6: Quiz Hook

**Assessment targets (not full questions — hooks only):**

1. **LoRA mechanism:** Given a weight matrix of dimension (2048, 2048) and LoRA rank 8, compute the number of trainable parameters. Compare to full fine-tuning. *[Tests: parameter counting, rank decomposition]*

2. **ControlNet injection point:** The ControlNet encoder output is added as a residual to which part of the U-Net? (encoder / decoder / both / bottleneck). Explain what would break if it were added to the encoder instead. *[Tests: architectural understanding of residual injection]*

3. **Conditioning tradeoff:** You increase the ControlNet conditioning scale from 0.5 to 1.5. Describe the expected behavior change in output fidelity vs. diversity. At what point do artifacts typically emerge? *[Tests: practical calibration knowledge]*

4. **Multi-adapter interaction:** You load two LoRAs — one trained on watercolor style (rank 32), one trained on photorealistic faces (rank 16). You merge both with scale 1.0. Predict whether the output will be a blend, dominant toward one style, or degenerate. Justify with rank-space reasoning. *[Tests: adapter composition theory]*

5. **GTM scenario:** A creative team generates 500 product images/week across 12 brands. Full fine-tuning per brand costs $200 GPU time. LoRA training costs $8. ControlNet eliminates manual layout adjustment (saves 15 min/image at $75/hr copywriter rate). Calculate the annual savings. *[Tests: quantitative GTM reasoning grounded in the mechanism]*

---

## Learning Objectives

1. **Compute** LoRA parameter counts given weight dimensions and rank, and compare to full fine-tuning costs
2. **Trace** the ControlNet data flow from spatial input through encoder to residual injection in the U-Net decoder
3. **Implement** a multi-adapter generation pipeline combining LoRA style weights with ControlNet spatial conditioning
4. **Calibrate** conditioning scales (guidance, controlnet, cross-attention) and predict their effect on output fidelity and diversity
5. **Evaluate** when parameter-efficient adaptation (LoRA) is economically justified over full fine-tuning for multi-brand visual content pipelines
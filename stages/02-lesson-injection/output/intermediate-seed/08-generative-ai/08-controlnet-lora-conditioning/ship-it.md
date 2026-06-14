## Ship It

In production, the stack gets more complex. A real brand asset pipeline layers multiple adapters and conditioning signals. Here is what a production configuration looks like for a creative team generating social media assets across a product catalog:

```python
import torch
from diffusers import StableDiffusionXLPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np
import cv2

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

pipe.load_lora_weights("brand-org/color-grade-lora", weight_name="color_v3.safetensors")
pipe.load_lora_weights("brand-org/product-style-lora", weight_name="style_v1.safetensors")
pipe.load_lora_weights("brand-org/quality-tuning-lora", weight_name="quality_v2.safetensors")

cross_attention_kwargs = {"scale": 1.0}
lora_scales = [0.7, 0.9, 0.5]

for i, (name, scale) in enumerate(zip(
    ["color_grade", "product_style", "quality"],
    lora_scales
)):
    pipe.fuse_lora(lora_scale=scale)
    print(f"Fused {name} at scale {scale}")

controlnet_canny = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.controlnet = controlnet_canny

product_image = load_image("product_base.jpg")
gray = cv2.cvtColor(np.array(product_image), cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 50, 150)
edge_map = load_image("layout_overlay.png")
edge_array = np.array(edge_map.convert("L"))
combined = np.maximum(edges, edge_array)
combined_image = load_image(
    cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    if isinstance(combined, np.ndarray) else combined
)

prompts = [
    "professional product photography on marble surface, soft shadows",
    "product hero shot, studio lighting, shallow depth of field",
    "lifestyle product image, natural window light, warm tones",
]

variants = []
for prompt in prompts:
    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, distorted, watermark, text, low resolution",
        image=combined_image,
        controlnet_conditioning_scale=0.85,
        guidance_scale=8.0,
        num_inference_steps=35,
        width=1024,
        height=1024,
        num_images_per_prompt=2,
        cross_attention_kwargs=cross_attention_kwargs,
    ).images
    variants.extend(result)

for i, img in enumerate(variants):
    img.save(f"social_asset_{i:03d}.png")

print(f"Generated {len(variants)} brand-consistent variants.")
print(f"Layout enforced by ControlNet canny at scale 0.85.")
print(f"Brand identity from 3 stacked LoRAs (color=0.7, style=0.9, quality=0.5).")
```

Deployment considerations for this pipeline:

**Inference cost.** Each ControlNet adds a forward pass through the cloned encoder. Two ControlNets roughly doubles U-Net inference time. On an A10g GPU, a single SDXL image at 1024×1024 with 30 steps takes about 3 seconds without ControlNet and 5 seconds with one ControlNet. Batch generation amortizes this — generating 4 images per forward pass costs only marginally more than 1. [CITATION NEEDED — concept: SDXL inference latency benchmarks with ControlNet on common GPU hardware]

**LoRA fusion vs. dynamic loading.** `fuse_lora()` permanently merges the adapter delta into the base weights in memory. This is faster at inference (no adapter overhead per forward pass) but prevents swapping adapters without reloading the model. For multi-brand pipelines where you generate for Brand A, then Brand B, keep adapters unfused and pass `cross_attention_kwargs={"scale": 0.8}` per call. The overhead is small (5-10% slower) and you gain flexibility. [CITATION NEEDED — concept: fused vs unfused LoRA inference latency comparison]

**ControlNet scale calibration.** The conditioning scale is not calibrated across ControlNet types. A Canny ControlNet at scale 0.8 behaves differently from a depth ControlNet at 0.8 — Canny enforces edges precisely, depth is softer. Always calibrate on a test set of 10-20 representative images before deploying to batch generation. The pattern: generate at scales [0.3, 0.5, 0.7, 0.9, 1.2], have a reviewer pick the value that enforces structure without artifacts, then lock that value for production.

**Storage and versioning.** A production brand pipeline accumulates artifacts: base model (7 GB), LoRA adapters (30-150 MB each), ControlNet checkpoints (1.5-5 GB each). Use a model registry (HuggingFace Hub, MLflow, or a simple S3 + manifest) and tag every artifact with brand ID, version, and training data hash. The LoRA file for "Brand A, style v3, trained on 2024 catalog" should be traceable to its training set and hyperparameters.
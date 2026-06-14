# Inpainting, Outpainting & Image Editing

## Learning Objectives

1. Implement mask-based inpainting by constructing binary masks and conditioning diffusion models on unmasked pixels.
2. Compare inpainting, img2img (SDEdit), and instruction-based editing along the axes of spatial control, coherence, and compute cost.
3. Generate outpainting by extending canvas dimensions and applying inpainting at boundary regions.
4. Build an automated mask-generation pipeline using color thresholding and contour detection.
5. Evaluate which editing paradigm fits a given task based on the ratio of preserved versus generated content.

## The Problem

A client sends a product photo that is 95% usable. The lighting is correct, the angle is correct, the background texture is correct — but there is a fire extinguisher mounted on the wall behind the product, and it has to go. If you run text-to-image from scratch with a prompt describing the scene, you get a new image. Different pixels everywhere. Different product angle. Different shadow. The client does not want a new image. They want *that* image, minus the fire extinguisher.

This is the constraint that defines image editing versus image generation: preserve a subset of pixels exactly, regenerate the rest coherently. The ratio of preserved to generated content determines which technique you reach for. Remove a small object from a large scene — that is inpainting, heavy on preservation. Change the entire style of a photo — that is img2img, heavy on regeneration. Extend a landscape image to a wider aspect ratio — that is outpainting, generating new content that is constrained to be continuous with the existing boundary.

In production, editing dominates generation. Most billable image work is modifying existing assets, not creating new ones from prompts. Swap a background, remove a logo, fix a distorted hand, extend a canvas for a different aspect ratio. Every major diffusion model shipped in the last two years includes a dedicated inpainting mode: Flux.1-Fill, Stable Diffusion Inpainting, SDXL Inpaint, DALL-E Edit. They all implement the same core mechanism — partial conditioning — with different architectures and quality tradeoffs that we will trace through in detail.

## The Concept

All three editing paradigms — inpainting, outpainting, and img2img — share one underlying principle: **partial conditioning**. The model receives the original image, a specification of which pixels to regenerate, and guidance for the new content. The unmasked pixels are clamped during denoising. The masked pixels follow the standard reverse diffusion trajectory. The surrounding context flows into the masked region through the U-Net's self-attention layers, producing a fill that is spatially coherent with the neighborhood.

The naive approach to inpainting — and the one most beginners try first — is to run standard text-to-image and then paste the result into the masked region of the original. This produces visible seams. The boundary between generated and original pixels does not match because the model never saw the surrounding context during generation. A slightly less naive version forward-diffuses the original image and injects the unmasked noisy latents at each denoising step. This reduces seams but still produces poor results because the standard U-Net has no mechanism to distinguish "region to generate" from "region to preserve." It treats all input channels uniformly.

The proper solution is a purpose-trained inpainting model. The U-Net is modified to accept 9 input channels instead of the standard 4: the 4-channel noisy latent, the 4-channel VAE-encoded original image, and a 1-channel binary mask. During training, the model learns to generate content in the masked region while respecting the unmasked context provided through the additional channels. This is why `runwayml/stable-diffusion-inpainting` and `stabilityai/stable-diffusion-xl-base-1.0` (in inpainting mode) produce coherent fills where naive paste-and-blend fails — the architecture itself is conditioned on the mask.

```mermaid
flowchart TD
    A[Original Image] --> B[VAE Encode → 4ch latent]
    C[Binary Mask] --> D[1ch mask tensor]
    E[Noise Latent] --> F[4ch noisy latent]
    B --> G
    D --> G
    F --> G[Concat → 9ch input]
    G --> H[Inpainting U-Net]
    I[Text Prompt → CLIP] --> H
    H --> J[Denoised Latent]
    J --> K[VAE Decode]
    K --> L[Output Image]
    L --> M[Composite: unmasked pixels from original]
    M --> N[Final Result]

    style G fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
```

The flow above shows what happens at each denoising step. The original image is VAE-encoded once, the mask is constant, and the noisy latent evolves through the timestep loop. The 9-channel concatenation happens inside the forward pass — the U-Net's first convolution layer expects 9 input channels and was trained with that dimensionality.

**img2img (SDEdit)** takes a different approach. Instead of adding noise only to masked pixels, it adds partial noise to the *entire* image, controlled by a `strength` parameter (0.0 means no change, 1.0 means full regeneration to noise). The model then denoises with a new prompt. This is useful for style transfer — "make this photo look like an oil painting" — because you want global modification, not surgical removal. It is poor for tasks like "remove the sign in the background" because every pixel is a candidate for change, including the product you want to preserve.

**Outpainting** is inpainting applied at image boundaries. You expand the canvas by padding with zeros, create a mask covering the padded region, and run the inpainting pipeline. The model extends edges using context from the original image. This works well when the boundary region contains continuous textures — skies, walls, landscapes — and poorly when the boundary cuts through a subject. Extending a landscape photo to widescreen is reliable. Extending a portrait to show more of the body is not, because the model has to hallucinate anatomy it cannot see.

**Instruction-based editing** (InstructPix2Pix) removes the mask requirement entirely. A diffusion model is fine-tuned on `(image, instruction, output)` triplets, learning to map natural-language commands like "make it snowy" or "remove the car" directly to edited images. This trades spatial precision for convenience — you cannot say "remove only the left car" without a mask — but it eliminates the masking step entirely, which is significant when you are processing thousands of images in a batch pipeline. [CITATION NEEDED — concept: InstructPix2Pix training dataset composition and instruction fidelity benchmarks]

## Build It

We start with mask generation, because without a mask, you have no inpainting. The most reliable automated approach for product photography and green-screen-style content is color thresholding: define a target color range in HSV space, classify each pixel as inside or outside that range, and produce a binary mask. This runs in milliseconds, requires no model download, and works on any image with a dominant background color.

```python
import numpy as np
from PIL import Image

def create_synthetic_product_image():
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:] = [34, 177, 76]
    cv2_region = img.copy()
    img[80:180, 100:160] = [200, 50, 30]
    img[90:170, 110:150] = [220, 220, 220]
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def color_threshold_mask(image_rgb, target_rgb, tolerance=30):
    diff = np.abs(image_rgb.astype(np.int16) - np.array(target_rgb, dtype=np.int16))
    distance = np.sqrt(np.sum(diff ** 2, axis=-1))
    mask = (distance < tolerance).astype(np.uint8) * 255
    return mask

product_img = create_synthetic_product_image()
Image.fromarray(product_img).save("product_photo.png")

mask = color_threshold_mask(product_img, target_rgb=[34, 177, 76], tolerance=40)

mask_pixels = np.count_nonzero(mask)
total_pixels = mask.shape[0] * mask.shape[1]
coverage = mask_pixels / total_pixels * 100

print(f"Image shape: {product_img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Masked pixels (background): {mask_pixels}")
print(f"Total pixels: {total_pixels}")
print(f"Background coverage: {coverage:.1f}%")
print(f"Foreground pixels (product): {total_pixels - mask_pixels}")

Image.fromarray(mask).save("background_mask.png")
print("Saved: product_photo.png, background_mask.png")
```

This produces a binary mask where the green background is white (255) and the product region is black (0). In an inpainting pipeline, you would invert this mask — you want to regenerate the background (remove it) while preserving the product. The tolerance parameter controls how strict the color match is. Too low, and shadows on the green screen are excluded from the mask. Too high, and green-tinted pixels in the product are incorrectly masked.

Now we run actual inpainting using a diffusion model. This requires the `diffusers` and `torch` libraries. The pipeline takes three inputs: the original image, the mask, and a text prompt describing what should fill the masked region.

```python
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw
import numpy as np

pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

source = Image.new("RGB", (512, 512), color=(68, 154, 79))
draw = ImageDraw.Draw(source)
draw.rectangle([180, 120, 332, 392], fill=(180, 60, 40), outline=(40, 40, 40), width=3)
draw.ellipse([220, 160, 292, 240], fill=(240, 220, 180))
draw.rectangle([240, 280, 272, 360], fill=(60, 60, 60))

mask = Image.new("L", (512, 512), 0)
mask_draw = ImageDraw.Draw(mask)
mask_draw.rectangle([100, 100, 250, 250], fill=255)

source.save("inpaint_source.png")
mask.save("inpaint_mask.png")

result = pipe(
    prompt="clean white office wall, professional photography, soft lighting",
    image=source,
    mask_image=mask,
    num_inference_steps=25,
    strength=0.85,
    guidance_scale=7.5,
).images[0]

result.save("inpaint_result.png")

source_arr = np.array(source)
result_arr = np.array(result)
mask_arr = np.array(mask) > 0

changed = np.any(source_arr[mask_arr] != result_arr[mask_arr])
unchanged_region = np.all(source_arr[~mask_arr] == result_arr[~mask_arr])

print(f"Source image size: {source.size}")
print(f"Mask coverage: {mask_arr.mean()*100:.1f}% of pixels")
print(f"Masked region changed: {changed}")
print(f"Unmasked region preserved exactly: {unchanged_region}")
print(f"Model device: {next(pipe.unet.parameters()).device}")
print("Saved: inpaint_source.png, inpaint_mask.png, inpaint_result.png")
```

The output confirms the core contract of inpainting: pixels inside the mask changed, pixels outside the mask did not. The `strength` parameter at 0.85 means the masked region is denoised starting from 85% noise — high enough to allow substantial regeneration, low enough to preserve coarse structure. For object removal, values between 0.8 and 1.0 are standard. For subtle modifications (changing a shirt color while keeping folds and shadows), 0.5–0.7 is better.

For outpainting, we extend the canvas and create a mask that covers only the new padding region:

```python
import numpy as np
from PIL import Image

def outpaint(image, padding_top, padding_bottom, padding_left, padding_right, pipe=None, prompt=""):
    orig_w, orig_h = image.size
    new_w = orig_w + padding_left + padding_right
    new_h = orig_h + padding_top + padding_bottom

    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    canvas.paste(image, (padding_left, padding_top))

    mask = Image.new("L", (new_w, new_h), 0)
    mask_arr = np.array(mask)
    mask_arr[:padding_top, :] = 255
    mask_arr[-padding_bottom:, :] = 255
    mask_arr[:, :padding_left] = 255
    mask_arr[:, -padding_right:] = 255
    mask = Image.fromarray(mask_arr)

    return canvas, mask

source = Image.new("RGB", (512, 512), color=(135, 206, 235))
from PIL import ImageDraw
draw = ImageDraw.Draw(source)
draw.ellipse([200, 200, 312, 312], fill=(255, 255, 100))
draw.rectangle([150, 350, 362, 512], fill=(80, 140, 60))

canvas, mask = outpaint(
    source,
    padding_top=128,
    padding_bottom=128,
    padding_left=128,
    padding_right=0,
)

canvas.save("outpaint_canvas.png")
mask.save("outpaint_mask.png")

original_coverage = (512 * 512) / canvas.size[0] / canvas.size[1] * 100
generated_coverage = 100 - original_coverage

print(f"Original size: {source.size}")
print(f"Canvas size: {canvas.size}")
print(f"Original content preserved: {original_coverage:.1f}%")
print(f"New pixels to generate: {generated_coverage:.1f}%")
print(f"Mask sum (white pixels): {np.array(mask).sum() // 255}")
print("Saved: outpaint_canvas.png, outpaint_mask.png")
```

The outpainting mask marks only the padding region as white. When you pass this canvas and mask into the inpainting pipeline, the model extends the sky and ground into the padded areas. The boundary where original meets padding is where coherence matters most — if the original image has a hard vertical edge at the boundary (like a building), the model will try to extend it, sometimes successfully, sometimes with artifacts.

## Use It

The partial conditioning mechanism — clamp a known-good subset, regenerate only the masked remainder with context awareness — maps directly to selective CRM enrichment (Cluster 1.3, TAM Refinement & ICP Scoring). Your CRM is a retrieval system whose value degrades when records contain stale or missing fields. Running a full enrichment waterfall on every record wastes API budget regenerating data that is already correct. Field-level masking is inpainting applied to structured data: preserve the trustworthy fields, flag only the stale and null ones for regeneration.

```python
import datetime

records = [
    {"id": "001", "domain": "acme.com", "industry": "SaaS",
     "employees": 250, "revenue": None, "updated": "2025-01-15"},
    {"id": "002", "domain": "globex.io", "industry": None,
     "employees": None, "revenue": 5_000_000, "updated": "2024-06-01"},
    {"id": "003", "domain": "initech.com", "industry": "Fintech",
     "employees": 50, "revenue": 2_000_000, "updated": "2025-11-01"},
]

DECAY_DAYS = 90
COST_PER_FIELD = 0.03
now = datetime.datetime.now()
enrichment_fields = ["industry", "employees", "revenue"]

mask = {}
for r in records:
    age = (now - datetime.datetime.strptime(r["updated"], "%Y-%m-%d")).days
    mask[r["id"]] = {f: (1 if r[f] is None or age > DECAY_DAYS else 0)
                     for f in enrichment_fields}

masked_updates = sum(sum(m.values()) for m in mask.values())
total_slots = len(records) * len(enrichment_fields)
full_cost = total_slots * COST_PER_FIELD
masked_cost = masked_updates * COST_PER_FIELD

print(f"Full enrichment:   ${full_cost:.2f}  ({total_slots} field lookups)")
print(f"Masked enrichment: ${masked_cost:.2f}  ({masked_updates} field lookups)")
print(f"Savings:           ${full_cost - masked_cost:.2f}  ({(1 - masked_cost/full_cost)*100:.0f}%)")
for rid, m in mask.items():
    flagged = [k for k, v in m.items() if v == 1]
    print(f"  {rid}: enrich {flagged or '— skip, all fields current'}")
```

Record 003 was updated recently with no nulls — its mask is all zeros, and the waterfall skips it entirely. Record 002 has nulls and is six months stale — every enrichment field is flagged. This is the data-pipeline analog of a binary inpainting mask: you define which fields need regeneration, clamp the rest, and spend API budget only on the masked subset.

The `denoising_strength` parameter has a structural analog here too. A low denoising strength (0.3–0.5) means "mostly preserve the original, make a small change." In CRM enrichment, this maps to field-level confidence thresholds: if an enrichment provider returns 247 employees for a record that previously said 250, the delta is small — accept the update silently. If it returns 2,470, the delta is large — flag it for human review before overwriting. The magnitude of change should govern the level of scrutiny applied.

## Exercises

**Exercise 1 — Mask sensitivity analysis.** Take the color thresholding function from Build It and run it on the synthetic product image with tolerance values of 10, 20, 30, 40, 50, 60. For each tolerance, print the mask coverage percentage and the number of foreground pixels incorrectly included in the background mask. Plot or print the tolerance-versus-accuracy tradeoff. Which tolerance value gives the cleanest separation?

**Exercise 2 — Denoising strength sweep.** Using the inpainting pipeline from Build It, run the same source image and mask with `strength` values of 0.3, 0.5, 0.7, 0.85, and 1.0. Save each result. For each output, compute the L2 pixel distance between the masked region of the output and the masked region of the original source. Print the distance for each strength value. At what strength does the model stop preserving any structural information from the original masked region? How does this threshold compare between object-removal prompts ("clean wall") versus modification prompts ("red car")?

## Key Terms

- **Inpainting** — Regenerating pixels inside a user-defined mask while clamping all unmasked pixels to their original values. Requires a 9-channel U-Net (4 noisy latent + 4 encoded original + 1 mask) or equivalent conditioning mechanism.
- **Outpainting** — A special case of inpainting where the mask covers only the padding region of an expanded canvas. The model extends boundary textures into the new area.
- **Img2img (SDEdit)** — Adding partial noise to the entire image (controlled by `strength`), then denoising with a new prompt. No mask is used; every pixel is a candidate for modification.
- **Partial conditioning** — The shared principle behind all editing paradigms: the model receives both the original content and a specification of what to change, and clamps the unmarked portion during denoising.
- **Denoising strength** — A float in [0.0, 1.0] controlling how much noise is added before denoising begins. Higher values allow more structural change; lower values preserve more of the original.
- **Binary mask** — A single-channel image where white (255) marks pixels to regenerate and black (0) marks pixels to preserve. The input that makes inpainting spatially precise.
- **Instruction-based editing** — A diffusion model fine-tuned on `(image, instruction, output)` triplets that applies edits from natural-language commands without requiring an explicit mask. Trades spatial precision for convenience.

## Sources

- Rombach, R., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." (CVPR 2022) — introduces the LDM architecture underlying Stable Diffusion, including the VAE-encode/decode and U-Net denoising loop that inpainting extends. https://arxiv.org/abs/2112.10752
- Meng, C., et al. "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations." (ICLR 2022) — defines the img2img / strength-based partial-noise approach. https://arxiv.org/abs/2108.01073
- Avrahami, O., et al. "InstructPix2Pix: Learning to Follow Image Editing Instructions." (CVPR 2023) — introduces instruction-based editing without explicit masks. [CITATION NEEDED — concept: InstructPix2Pix training dataset composition and instruction fidelity benchmarks]
- Saharia, C., et al. "Palette: Image-to-Image Diffusion Models." (SIGGRAPH 2022) — conditions diffusion on source image and mask via channel concatenation, the architectural pattern used by SD Inpainting. https://arxiv.org/abs/2111.05805
- Hugging Face Diffusers documentation — `AutoPipelineForInpainting` API and the 9-channel input contract for SD/SDXL inpainting models. https://huggingface.co/docs/diffusers/en/api/pipelines/auto_pipeline
- [CITATION NEEDED — concept: CRM field-level decay rates and selective enrichment cost benchmarks]
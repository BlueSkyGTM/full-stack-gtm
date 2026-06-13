# Inpainting, Outpainting & Image Editing

## Learning Objectives

1. Implement mask-based inpainting by constructing binary masks and conditioning diffusion models on unmasked pixels.
2. Compare inpainting, img2img (SDEdit), and instruction-based editing along the axes of control, coherence, and compute cost.
3. Generate outpainting by extending canvas dimensions and applying inpainting at boundary regions.
4. Build an automated mask-generation pipeline using color thresholding and contour detection.
5. Evaluate which editing paradigm fits a given task based on the ratio of preserved vs. generated content.

---

## Beat 1: Hook

Image generation from scratch is solved. The harder problem: edit one thing in an existing image without breaking everything else. That constraint—preserve X, change Y—is what separates inpainting from unconditional generation, and it's the mechanism behind every "remove this object" and "extend this background" tool on the market.

---

## Beat 2: Concept

Three editing paradigms, one underlying principle: **partial conditioning**. The model receives the original image, a binary mask indicating which pixels to regenerate, and a text prompt guiding the new content. Unmasked pixels are clamped during denoising; masked pixels follow the standard reverse diffusion trajectory.

**Inpainting**: Mask a region (object, face, text), run denoising only inside the mask. The model sees the surrounding context via self-attention and generates coherent fill. Critical parameter: `denoising_strength` controls how freely the model ignores the original pixels in the masked area.

**Outpainting**: Inpainting applied at image boundaries. You expand the canvas (padding), mask the new empty pixels, and inpaint. The model extends edges using context from the original image. Works best when the boundary contains continuous textures (skies, walls, landscapes) rather than sharp subject boundaries.

**img2img / SDEdit**: Add partial noise to the entire image (controlled by `strength` parameter, 0.0 = no change, 1.0 = full regeneration), then denoise with a new prompt. Not mask-based—every pixel is a candidate for change. Useful for style transfer and global edits; poor for surgical changes.

**Instruction-based editing** (InstructPix2Pix architecture): A diffusion model trained on `(image, instruction, output)` triplets. Takes natural language instructions ("make it snowy", "remove the car") instead of requiring a mask. Trades spatial control for convenience. [CITATION NEEDED — concept: InstructPix2Pix training dataset composition and instruction fidelity benchmarks]

**Mask generation approaches**: Hand-drawn masks (manual), color-based thresholding (detect a colored region), semantic segmentation masks (detect all "person" pixels), and depth/normal map thresholds (detect foreground objects).

---

## Beat 3: Demonstration

**Demo 1 — Mask generation from color threshold**: Load an image with a solid-color region (e.g., green screen background). Convert to HSV, threshold on hue range, produce a binary mask. Print mask statistics (percentage of image masked, bounding box coordinates).

**Demo 2 — Inpainting via API call**: Send the image + mask + prompt to a diffusion inpainting endpoint (Stability AI or replicate hosted Stable Diffusion Inpainting). Save the result. Show the masked region before/after.

**Demo 3 — img2img comparison**: Take the same image, run img2img at strength 0.3 and 0.7 with the same prompt. Show how global the changes are vs. the surgical inpainting result.

**Demo 4 — Outpainting**: Take a 512×512 image, pad to 768×768 with black pixels, create a mask of the padded region, inpaint with a context-appropriate prompt. Save the extended image.

Each demo produces saved images and printed statistics (mask coverage, latency, token usage where applicable).

---

## Beat 4: Use It

**GTM Redirect — Zone 2 Enrichment: Creative Asset Personalization for ABM**

When running account-based campaigns, personalizing imagery per account (logo swaps, background text changes, region-specific imagery) is currently manual design work. Inpainting automates the "replace this region with that content" step. A pipeline that: (1) holds a template image, (2) masks the personalization region, (3) inpaints with account-specific prompts, (4) outputs per-account variants—is a direct application in creative operations.

This is not a Clay waterfall pattern. It is a batch image-processing pipeline that feeds into campaign tools (HubSpot, Customer.io) as personalized creative assets. The mechanism is mask-conditioned diffusion applied repeatedly over a template.

**Exercise hooks:**
- *Easy*: Generate a mask for the center 25% of a template image and inpaint with two different prompts. Compare outputs.
- *Medium*: Build a function that takes a list of company names and a template image, masks the "logo area," and produces one inpainted variant per company.
- *Hard*: Construct an end-to-end pipeline: read account data from a CSV, generate personalized images via inpainting, and write output paths back to the CSV. Measure per-image cost and latency.

---

## Beat 5: Ship It

**Deliverable**: A CLI tool that accepts an input image, a mask (generated via color threshold or provided as a file), and a prompt. It outputs the inpainted result. Supports three modes: `inpaint`, `outpaint` (with configurable padding), and `img2img` (with configurable strength). Prints a JSON summary of the operation (mode, mask percentage, latency, cost).

**Exercise hooks:**
- *Easy*: Implement the `inpaint` mode with a provided mask file and prompt. Verify the masked region changed and the unmasked region is pixel-identical.
- *Medium*: Add the `outpaint` mode with automatic canvas expansion and mask generation. Handle edge cases where the prompt describes content inconsistent with the source image boundary.
- *Hard*: Add `img2img` mode and a `--compare` flag that runs both inpainting and img2img on the same image/mask/prompt and saves both outputs side by side for manual quality review.

---

## Beat 6: Review

**Key distinctions to internalize:**

| Paradigm | Spatial Control | Prompt Dependence | Best For |
|---|---|---|---|
| Inpainting | High (mask-defined) | Medium (guides masked region) | Object removal, region replacement |
| Outpainting | Medium (boundary-defined) | Medium (must extend context) | Canvas extension, aspect ratio changes |
| img2img | Low (global) | High (drives full regeneration) | Style transfer, global mood changes |
| Instruction-based | None (model decides) | High (instruction is primary input) | Quick edits when masking is impractical |

**Failure modes to watch for:**
- **Boundary seams**: Inpainting at `denoising_strength` too low produces visible seams at mask edges. The generated pixels don't blend with preserved pixels.
- **Context bleed**: Outpainting where the model "copies" boundary objects into the extended region instead of continuing the background.
- **Over-editing in img2img**: At strength > 0.5, semantic content shifts unpredictably. The prompt overwrites the original composition.

**One mechanism to remember**: All four paradigms are the same diffusion process with different conditioning constraints. The mask is just a per-pixel switch between "clamp to original" and "denoise freely." There is no separate "inpainting model"—only a model fine-tuned on masked training examples to handle the boundary coherence problem better than a base model would.

---

## GTM Redirect Rules Summary

- **Primary cluster**: Zone 2 Enrichment — creative asset personalization for ABM campaigns
- **Mechanism link**: Mask-conditioned diffusion applied as a batch operation over account-specific prompts
- **Honest boundary**: This is not a data-enrichment waterfall or a prospecting tool. It is a batch media generation step that feeds into campaign execution. If the student does not work with visual creative assets, this lesson is foundational for Zone 3 (Outreach) creative personalization patterns.
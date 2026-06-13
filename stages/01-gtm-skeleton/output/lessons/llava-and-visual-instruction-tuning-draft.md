# LLaVA and Visual Instruction Tuning

## Hook It

LLaVA proved you don't need a bespoke multimodal architecture—just bolt a vision encoder to an existing LLM via a projection layer, then instruction-tune the result. The insight that changed everything: GPT-4 can *generate* your visual instruction data from image captions alone, no human labeling required.

## Ground It

Prerequisites: contrastive learning (CLIP), causal language modeling, instruction tuning, and the concept of a projection/adapter layer between frozen encoder and frozen decoder. Assumes comfort with HuggingFace `transformers` pipeline APIs.

## Explain It

Mechanism breakdown: (1) CLIP ViT encodes image patches into visual tokens. (2) A trainable projection layer maps those tokens into the LLM's embedding space—treating them like additional text tokens. (3) Two-stage training: first align projections on image-caption pairs (CC3M filtered to ~595K), then visual instruction tuning on ~80K GPT-4-generated conversations where GPT-4 hallucinates plausible Q&A from captions alone. The LLM backbone stays frozen in stage 1, then full model fine-tunes in stage 2. Compare linear projection vs. two-layer MLP. Show that LLaVA's innovation is the *data generation pipeline*, not the architecture.

## Use It

**GTM Redirect → Zone 2 / Enrichment.** Visual instruction tuning lets you build enrichment agents that read screenshots, logos, PDFs, and webpage captures—extracting account signals no text scraper can reach. Exercise hook (easy): load a LLaVA checkpoint and run inference on a company screenshot to extract ICP-relevant attributes. Exercise hook (medium): batch-process a directory of competitor landing-page screenshots and output structured JSON per page. Exercise hook (hard): construct a custom visual-instruction dataset from your own screenshot+caption pairs, then fine-tune the projection layer for domain-specific extraction.

## Ship It

Serving considerations: LLaVA's inference path concatenates vision tokens with text tokens before the full transformer forward pass, so memory scales with image resolution and prompt length. Quantization (bitsandbytes 4-bit) drops VRAM to ~6 GB for the 7B variant. Batch image preprocessing through CLIP separately to avoid re-encoding the same image across multiple prompts. Exercise hook (easy): serialize a LLaVA pipeline with 4-bit quantization and confirm VRAM usage prints under threshold.

## Push It

Open research edges: hallucination rates on visual Q&A remain high—LLMs confabulate details not present in the image. The projection layer is a bottleneck; newer work (LLaVA-NeXT, LLaVA-OneVision) patches this with dynamic high-resolution slicing and multi-crop encoding. [CITATION NEEDED — concept: benchmark suite for multimodal hallucination detection in production]. Exercise hook (hard): run LLaVA and GPT-4V on the same 20 images, diff their outputs, and classify error types (hallucination, omission, misattribution) into a structured report.

---

**Learning Objectives (draft for `docs/en.md`):**

1. Implement a CLIP-ViT → projection → LLM forward pass and print the shape of the visual token tensor at each stage.
2. Compare linear vs. MLP projection layers by fine-tuning each on a held-out image-caption subset and reporting accuracy delta.
3. Construct a visual-instruction dataset from raw image captions using a language-only LLM (GPT-4 or open equivalent) to generate conversation, detail, and complex-reasoning splits.
4. Detect and classify hallucination types in LLaVA outputs against a ground-truth image-description baseline.
5. Configure LLaVA inference with 4-bit quantization and confirm VRAM consumption stays below a stated threshold on a single GPU.
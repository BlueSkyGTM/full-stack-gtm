# Vision-Language Models — The ViT-MLP-LLM Pattern

## Hook It

The modern vision-language model isn't a single monolith. It's three components bolted together: a ViT that turns pixels into tokens, an MLP projector that reshapes those tokens to match the LLM's embedding dimension, and the LLM itself that reasons over both image and text tokens. Every open-weight multimodal model—LLaVA, Qwen-VL, InternVL—uses some variant of this pattern. If you can't name the three stages, you can't debug when the pipeline fails.

## Ground It

**Prerequisites (and where to find them):**
- Transformer attention mechanism (Lesson: Transformer Architecture — Attention Is a Dot Product)
- CLIP dual-encoder pattern (Lesson: Contrastive Learning — The CLIP Pattern)
- Tokenization and embedding lookup (Lesson: Tokenization — BPE and Beyond)

**Key terms (no prior lesson assumed):**
- **Patch embedding**: slicing an image into fixed-size grids (e.g., 14×14 px), projecting each into a vector—exactly how a text tokenizer slices strings into subword tokens
- **Visual projector / adapter**: a learned linear or MLP layer that maps ViT output dimension to LLM input dimension
- **Cross-modal token concatenation**: prepending or interleaving projected visual tokens with text tokens before feeding the combined sequence to the LLM

**Mechanism first, tools second:**
The mechanism is token-space alignment: two independently trained encoders produce vectors in different dimensions. The MLP projector learns a linear-ish mapping from ViT-space to LLM-space. No shared training required at alignment time—the projector is the only component trained in many fine-tuning setups.

## Explain It

**The pattern, end to end:**

```
Image → Patch Extractor → ViT Encoder → Visual Tokens (N × d_vit)
                                              ↓
                                         MLP Projector
                                              ↓
                                     Projected Tokens (N' × d_llm)
                                              ↓
[PROJECTED_TOKENS] + [TEXT_TOKENS] → LLM → Output Text
```

**Component-by-component:**

1. **ViT Encoder**: Takes a 224×224 image, slices it into 14×14 patches (256 patches), each patch gets projected to d_vit dimensions. 16+ transformer layers of self-attention over those 256 patch tokens. Output: sequence of visual token vectors.

2. **MLP Projector**: The bottleneck. Input dimension is d_vit (e.g., 1024 for ViT-L). Output dimension must match d_llm (e.g., 4096 for Llama-7B). Usually 2-layer MLP with GELU. This is the only component trained from scratch in the original LLaVA training recipe—ViT and LLM weights stay frozen.

3. **LLM Decoder**: Receives `[VISUAL_TOKENS] + [TEXT_TOKENS]` as a single sequence. Standard causal attention over the concatenated input. The LLM has no special "vision" mechanism—it just sees more tokens at the front of the sequence.

**Why this works (and where it breaks):**
- Works because both modalities are now in the same token space—the LLM can't tell which tokens came from pixels vs. text
- Breaks when the projector undertrains (visual tokens look like noise to the LLM) or when the ViT hasn't seen similar images in pretraining
- The "hallucination" problem in VLMs often traces back to the projector passing blurry or ambiguous visual features

**Tools that implement this pattern:**
- **LLaVA** (open-source): the canonical ViT-MLP-LLM implementation. ViT-L/14 from CLIP + MLP + Vicuna/Llama
- **Qwen-VL**: same pattern, different ViT (ViT-bigG), adds visual attention abstraction to compress token count
- **InternVL**: swaps the MLP for a more complex dynamic compressor, but the three-stage pipeline is identical

**What's observable vs. claimed:**
- Observable: you can extract intermediate outputs at each stage and verify dimensions
- Observable: projector weights change during fine-tuning, ViT/LLM weights don't (in standard LLaVA training)
- Claimed but hard to verify: "the model truly understands spatial relationships"—often fails on counting, spatial reasoning, small text in images

## Use It

**GTM Cluster: Enrichment — Visual Signal Extraction**

Vision-language models map directly to enrichment workflows where the signal is visual, not textual:

- **Competitive intelligence**: screenshot competitor landing pages, extract positioning claims, pricing tiers, and feature lists from the rendered page (not the DOM—the actual visual)
- **Account scoring from visual assets**: ingest prospect logos, social media images, pitch deck screenshots to detect company stage, industry, and brand maturity
- **Personalization triggers**: analyze a prospect's LinkedIn banner image or website hero section to generate relevant icebreakers—this is the VLM equivalent of the Clay waterfall enrichment step, where a visual signal feeds the next enrichment node

**Where the redirect is honest:**
This is NOT "use VLM to write better emails." The mechanism is: VLM extracts structured data from unstructured visual input, and that structured data enters your existing enrichment waterfall. The VLM is one node in the pipeline, not the pipeline itself.

**Foundational note:** If your GTM workflow has zero visual inputs (no screenshots, no PDFs, no images), VLMs add no value over text-only LLMs. This is not a universal upgrade—it's a specific tool for a specific modality gap.

## Ship It

**Exercise hooks (not full text):**

**Easy — Verify the token pipeline dimensions:**
Load a pretrained LLaVA model (via `transformers`), feed it a single image, and print the tensor shape at three points: after ViT, after projector, after concatenation with text tokens. Confirm dimensions match the expected d_vit → d_llm mapping.

**Medium — Extract and inspect visual tokens:**
Run an image through the ViT + projector pipeline, extract the projected visual tokens, and compute cosine similarity between individual patch tokens. Identify which patches are most similar (likely background regions) vs. most distinct (likely foreground objects). Print the top-5 most similar pairs with their patch coordinates.

**Hard — Swap the projector, measure the degradation:**
Replace the trained MLP projector with a random linear layer of the same dimensions. Run the same prompt+image through the model and compare outputs. Quantify the degradation by measuring token-level divergence (logits KL divergence) between the trained-projector output and the random-projector output. This demonstrates what the projector actually contributes.

## Stretch It

**Where this pattern is going:**
- **Token compression**: Qwen-VL and InternVL compress 256+ visual tokens down to 32–64 before feeding the LLM. Reduces compute, but risks losing fine-grained detail (small text, distant objects). Tradeoff is unresolved.
- **Multiple images**: the basic pattern handles one image. Multi-image reasoning (comparing two screenshots, reading a multi-page document) requires interleaving multiple sets of visual tokens—context window becomes the bottleneck.
- **Video**: same pattern, but now you're feeding 8–32 frames × 256 tokens per frame = 2000+ visual tokens per second. Memory and attention scaling are open problems.

**Foundational for Zone 3 (AI-native product features):** If your product includes visual inputs—document processing, screenshot analysis, image-based search—the ViT-MLP-LLM pattern is the default architecture. Understanding it lets you debug latency (it's the ViT), hallucination (it's the projector or insufficient ViT pretraining), and cost (token count scales with image resolution, not image complexity).

**Open questions the practitioner should track:**
- Does token compression (256→64 tokens) lose information that matters for your specific GTM use case? Test empirically.
- Can you fine-tune only the projector on domain-specific images (your competitors' screenshots) and get measurable improvement? The original LLaVA paper suggests yes, but [CITATION NEEDED — concept: projector-only fine-tuning effectiveness on domain-specific images in production VLM deployments].
- Are text-only models with good OCR a cheaper alternative for your use case? If the visual content is mostly text (websites, slides), OCR + text LLM may outperform VLM at lower cost.
# Janus-Pro: Decoupled Encoders for Unified Multimodal Models

## Learning Objectives

- Explain why a single shared visual encoder compromises either understanding or generation quality, and trace the representational conflict to its source.
- Implement the dual-encoder routing pattern — SigLIP for understanding, VQ tokenizer for generation — and measure the representation distance between the two pathways on identical input.
- Run Janus-Pro-7B for both multimodal understanding (image → text) and image generation (text → image) inference.
- Compare the decoupled architecture of Janus-Pro against coupled-continuous (Transfusion) and coupled-discrete (Show-o) alternatives on the axes of parameter efficiency, representational conflict, and task quality.
- Wire multimodal understanding and generation outputs into a GTM enrichment and outreach pipeline with observability on encoder pathway behavior.

## The Problem

Unified multimodal models share a transformer body across understanding (image in, text out) and generation (text in, image out). The transformer body is fine with this — self-attention is task-agnostic. The problem lives one layer upstream: the visual encoder.

A visual encoder converts pixels into tokens the transformer can process. But understanding and generation want fundamentally different things from that encoding. Understanding wants semantic embeddings — vectors where all "logo on white background" images cluster near each other regardless of pixel-level differences. SigLIP and DINOv2 produce exactly this: high-level features aligned with text concepts. Generation wants reconstruction-friendly codes — tokens that carry enough spatial and detail information to be decoded back into crisp pixels. VQ tokenizers (VQ-VAE, VQGAN) produce this: discrete codes from a learned codebook that map to local image patches.

These two objectives are in direct tension. A semantic encoder throws away pixel-level detail to build concept-level abstraction — that's the whole point. A reconstruction encoder preserves pixel detail at the cost of semantic coherence. Previous unified models forced one encoder to serve both. Show-o and Transfusion both use a single visual tokenizer for both directions. The result is a visible quality tax: Show-o's understanding scores trail specialist models, and its generation quality is mid-tier. Transfusion avoids discrete tokenization for generation (using continuous diffusion instead) but still pays a representational cost on the understanding side because the shared image encoder is optimized for reconstruction.

The deeper issue is not engineering — it's information-theoretic. Semantic compression and pixel-level fidelity are competing objectives along the same channel. You cannot maximize both in a single representation. You can compromise on both, which is what coupled architectures do. Or you can run two channels, which is what Janus-Pro does.

## The Concept

Janus-Pro's core move is to stop fighting the representational conflict and route around it. Two independent visual encoders feed a shared transformer backbone. Each encoder is optimized for its own task without degrading the other.

**The understanding pathway.** When the task is "describe this image," the input image passes through a SigLIP encoder. SigLIP produces a set of feature vectors that capture semantic content — objects, scenes, relationships, text in the image. These features are projected into the LLM's embedding space via a linear adapter. The LLM receives them as a sequence of visual tokens interleaved with text tokens, and autoregressively generates a text response. This pathway is architecturally identical to LLaVA and other vision-language models.

**The generation pathway.** When the task is "generate an image from this text," no image passes through the SigLIP encoder. The text prompt is tokenized and fed into the LLM directly. The LLM autoregressively generates a sequence of discrete tokens — but instead of text vocabulary tokens, it generates indices into a VQ codebook. Each index corresponds to a learned code vector that represents a local image patch. Once enough tokens are generated (576 for Janus-Pro's 384×384 output), the VQ decoder maps the codebook indices back to pixels. This pathway resembles standard autoregressive image generation (DALL-E, Parti).

**The shared backbone.** The LLM (DeepSeek-LLM, a decoder-only transformer) processes tokens from either encoder through the same attention and feed-forward layers. The task determines which encoder feeds input and which head receives output. The backbone learns unified representations that accommodate both semantic and spatial tokens — but the encoders themselves never interfere.

```mermaid
flowchart TD
    UI[Image Input] --> SE[SigLIP Encoder<br/>Semantic Features]
    SE --> PA[Projection Adapter]
    PA --> LLM[Unified LLM Backbone<br/>DeepSeek-LLM]
    
    TP[Text Prompt] --> TT[Text Tokenizer]
    TT --> LLM
    
    LLM -->|"understanding task"| TH[Text Head → Decode]
    TH --> TO[Text Output]
    
    LLM -->|"generation task"| VQH[VQ Token Head<br/>576 tokens × codebook]
    VQH --> VQD[VQ Decoder]
    VQD --> GI[Generated Image]
    
    style SE fill:#e1f5fe
    style VQD fill:#fff3e0
    style LLM fill:#
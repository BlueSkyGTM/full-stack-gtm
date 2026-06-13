# Production Quantization — AWQ, GPTQ, GGUF K-quants, FP8, MXFP4/NVFP4

## GTM Redirect Rules

GTM Cluster 13 — Deployment, CI/CD → Production GTM Infrastructure. Quantized models run your Clay enrichment and n8n workflows on cheaper GPU hosts or local hardware. The redirect: "this quantized deployment is your GTM inference layer — the same pipeline that runs your Clay tables and classification agents at 3× cheaper hosting."

---

## Beat 1: Hook — Why This Now

A 7B model in FP16 needs 14 GB of VRAM just to load. Your prospect-classification agent does not need FP16 precision to return a JSON label. Quantization is the difference between an A100 and a T4 — between $2.50/hr and $0.40/hr for your GTM inference stack. Every production deployment you ship will face this constraint.

---

## Beat 2: Concept — The Quantization Taxonomy

Two axes define every quantization scheme: (1) what gets quantized — weights only, or weights and activations — and (2) how the scale factor is computed — per-tensor, per-channel, per-group, or per-block. Map the landscape: post-training quantization (PTQ) vs. quantization-aware training (QAQ), symmetric vs. asymmetric, and where AWQ, GPTQ, GGUF, FP8, and MXFP4 sit on this grid. The practitioner's mental model: quantization = `dequant(quant(x))` and the error term is the entire game.

---

## Beat 3: Mechanism — How Each Method Works

**GPTQ**: Approximates the Hessian of the layer-wise reconstruction error, quantizes weights column-by-column while adjusting remaining columns to compensate. Requires calibration dataset. Produces INT4/INT3 weights with groupwise scales.

**AWQ**: Observes that ~1% of weight channels are "salient" (high activation magnitude) and protecting those channels during quantization preserves accuracy better than uniform treatment. No Hessian — uses activation statistics from calibration data.

**GGUF K-quants** (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0): Super-block structure where each super-block has a single scale, and sub-blocks within it have their own scales. Different K-quant levels mix bit widths within the same tensor (e.g., Q4_K_M uses 4-bit for most weights, 6-bit for the most important). Designed for CPU/Apple Silicon inference via llama.cpp.

**FP8** (E4M3 for forward, E5M2 for backward): Hardware-native on H100/4090. Not integer quantization — a reduced-precision float with fewer mantissa/exponent bits. Delayed scaling: track max values over a window of tensors, update scale periodically.

**MXFP4/NVFP4**: Microscaling formats. Divide tensors into micro-blocks (e.g., 32 elements), share a single FP8 scale across the block, store elements in 4-bit. NVFP4 is NVIDIA's Blackwell variant with specific block size and scale format. [CITATION NEEDED — concept: MXFP4 micro-block size and scale encoding for Blackwell NVFP4]

Exercise hooks:
- **Easy**: Load a model in two different GGUF K-quant levels, run the same prompt, compare token/sec and perplexity.
- **Medium**: Run GPTQ and AWQ on the same model, measure perplexity delta on a held-out corpus.
- **Hard**: Implement per-group symmetric quantization from scratch in NumPy, reproduce the MSE floor.

---

## Beat 4: Use It — Quantize a Model for Your Inference Stack

Concrete walkthrough: quantize a 7B classifier model (the kind that labels inbound leads for your GTM pipeline) with AWQ for GPU serving and export to GGUF Q4_K_M for local/CPU fallback. Show the vLLM serving command for the AWQ model and the llama.cpp command for the GGUF model. Measure: latency P50/P99, throughput, memory footprint, and classification accuracy on a held-out set of 500 labeled leads.

GTM redirect: "This is your GTM inference layer. The quantized model classifies inbound signals for your Clay tables and n8n routing — at a hosting cost that doesn't eat your retainer margin."

Exercise hooks:
- **Easy**: Serve an AWQ-quantized model via vLLM, send 10 requests, log latency.
- **Medium**: Convert a model to GGUF Q4_K_M, serve via llama.cpp, compare latency to the AWQ version.
- **Hard**: Build a benchmark harness that sweeps Q2_K through Q8_0, plots accuracy-vs-latency Pareto frontier.

---

## Beat 5: Ship It — Deployment Checklist and Hardware Matching

Production deployment matrix: which quantization format matches which hardware. H100 → FP8 or MXFP4. T4/4090 → AWQ INT4. CPU/Apple Silicon → GGUF. Memory budget worksheet: parameter count × bytes per param × 1.2 (overhead). CI/CD integration: quantize in build pipeline, store artifact in registry, serve from canary. Monitoring: track inference error rates and log the quantization format alongside every prediction for post-hoc accuracy analysis.

GTM redirect: "This deploy pipeline ships your quantized models alongside your Clay tables and n8n workflows. Your SPF/DKIM/DMARC is infrastructure; your quantized inference endpoint is the same category — invisible until it breaks."

Exercise hooks:
- **Easy**: Write a Dockerfile that serves a GGUF model via llama.cpp server.
- **Medium**: Create a GitHub Actions workflow that quantizes a model on push and uploads the artifact.
- **Hard**: Deploy canary A/B between FP16 and AWQ, measure accuracy drift over 1000 real requests with logged ground-truth labels.

---

## Beat 6: Extend — Where the Floor Is Moving

FP8 is becoming the default training format, not just inference. MXFP4/NVFP4 will ship on Blackwell and change the breakeven calculus between "just use FP8" and "compress further." ONNX and TensorRT-LLM are converging on FP8 as the portable interchange format. The practical takeaway: if you are building inference infrastructure today, FP8-first with AWQ fallback covers 90% of deployments. GGUF remains the CPU/local champion. GPTQ is being displaced by AWQ in new deployments for accuracy reasons, but existing GPTQ artifacts in HuggingFace are abundant.

Reading: the AWQ paper (Lin et al. 2023), GPTQ paper (Frantar et al. 2022), GGUF spec in llama.cpp repository, NVIDIA FP8 spec, [CITATION NEEDED — concept: MXFP4 specification and block format reference].
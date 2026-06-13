# TensorRT-LLM on Blackwell with FP8 and NVFP4

---

## Beat 1: Hook

NVIDIA Blackwell introduces NVFP4 — a 4-bit floating-point format that cuts per-token memory bandwidth in half compared to FP8. TensorRT-LLM compiles models into TensorRT engines that exploit this format. The mechanical difference: FP8 uses E4M3/E5M2 encoding at 1 byte/param; NVFP4 uses E2M1 with a block-scale factor at 0.5 bytes/param. Half the bits, same floating-point structure, new scaling mechanics. If you deploy LLMs, this changes your cost-per-token arithmetic.

---

## Beat 2: Concept

**Mechanism: Block-scaled floating point.** Standard quantization maps weights to a uniform grid. Block-scaled FP4 divides weights into blocks (typically 16 elements), computes a shared scale per block, and stores each element as a 4-bit mantissa-exponent pair relative to that scale. This preserves dynamic range that integer quantization (INT4) destroys.

**Precision landscape on Blackwell:**
- FP16/BF16 — 2 bytes/param, baseline accuracy
- FP8 (E4M3) — 1 byte/param, ~1% perplexity degradation, available on Hopper
- NVFP4 (E2M1 + block scale) — 0.5 bytes/param, 2-5% perplexity degradation, Blackwell-only

**Key constraint:** NVFP4 requires calibration data. You cannot cast weights to FP4 and expect usable output — you must run representative data through the model first so TensorRT-LLM can compute per-block scales.

---

## Beat 3: Mechanism

**Step-by-step compilation pipeline:**

1. **Load HF checkpoint** → TensorRT-LLM reads HuggingFace weights
2. **Calibrate** → Forward pass on representative data, collect activation statistics
3. **Quantize** → Apply per-block FP4 scaling to weights, per-tensor scaling to activations
4. **Build engine** → TensorRT fuses kernels, optimizes memory layout for Blackwell's TMA units
5. **Run** → Sessions with inflight batching, paged KV cache

**Engine configuration knobs:**
- `quant_mode` — sets FP8 vs FP4
- `calib_dataset` — representative samples for scale computation
- `max_batch_size`, `max_seq_len` — pre-allocate KV cache
- `world_size`, `tp_size` — tensor parallelism across GPUs

**Code exercise (Easy):** Print available quantization modes for a given TensorRT-LLM version. Observable: enum values for FP8, FP4.

**Code exercise (Medium):** Write a calibration config that specifies dataset path, sequence length, and number of calibration samples. Print the config to stdout as JSON.

**Code exercise (Hard):** Given a pre-built FP8 engine and a pre-built FP4 engine (provided as artifacts), run inference on 50 prompts with both, measure tokens/second and VRAM usage, print comparison table.

---

## Beat 4: Use It

**GTM Redirect:** This is foundational for Zone 1 (AI Foundations). Inference cost per token determines whether AI-powered enrichment workflows — Clay waterfalls, automated research agents, personalized outreach generation — are economically viable at scale. FP4 quantization can halve GPU memory requirements, which either halves your compute bill or doubles your throughput on the same hardware.

**Practical application:** Build a TensorRT-LLM engine for a 7B-parameter model in FP4. Measure tokens/second against the FP16 baseline. Compute cost-per-1K-tokens at current GPU hourly rates.

**Code exercise (Medium):** Write a script that takes an HF model ID, runs the TensorRT-LLM build pipeline with FP4 quantization, and saves the engine to disk. Print build time and engine file size. (Assumes Blackwell GPU available.)

---

## Beat 5: Ship It

**Deployment concerns:**

- **Engine portability:** TensorRT engines are GPU-architecture specific. An FP4 engine built on Blackwell will not run on Hopper. You build once per target architecture.
- **Batching strategy:** Inflight batching (IFB) is required for production. TensorRT-LLM implements this; configure `max_num_tokens` to control the batching window.
- **KV cache management:** Paged attention with block-manager prevents fragmentation. Set `tokens_per_block` based on your sequence length distribution.
- **Accuracy validation:** Run perplexity evaluation (wikitext2, or domain-specific corpus) on the quantized engine. If perplexity increases >5%, fall back to FP8.

**GTM Redirect:** Same as Beat 4 — foundational for Zone 1. The shipping decision is a cost/accuracy tradeoff. If your AI enrichment pipeline tolerates 3-5% accuracy degradation, FP4 cuts your inference bill in half. If not, FP8 is the floor.

**Code exercise (Hard):** Write a benchmark harness that: (1) loads FP4 and FP16 engines, (2) runs 500 prompts through each, (3) computes tokens/second, median latency, p99 latency, and VRAM peak, (4) prints a markdown comparison table.

---

## Beat 6: Evaluate

**Learning Objectives:**

1. Configure TensorRT-LLM to compile a model with FP8 and NVFP4 quantization on Blackwell GPUs
2. Compare inference throughput and memory footprint between FP8 and NVFP4 precisions using benchmark harnesses
3. Implement calibration workflows for FP4 quantization using representative datasets
4. Diagnose precision-related accuracy degradation using perplexity benchmarks
5. Deploy a quantized TensorRT-LLM engine with inflight batching and paged KV cache

**Quiz hooks (not full questions):**
- Identify the correct encoding structure of NVFP4 (E2M1 + block scale vs. uniform quantization)
- Given two perplexity numbers (FP16 baseline, FP4 quantized), determine whether the degradation is within acceptable range
- Predict which GPU architectures support NVFP4 execution
- Explain why calibration data is required for FP4 but not for FP8 weight-only quantization
- Given a VRAM budget and model size, calculate whether FP4 fits while FP8 does not

---

## Notes

- **GPU availability:** All code exercises that build/run FP4 engines require Blackwell hardware (B100/B200). Exercises that only inspect configs or print quantization modes can run anywhere.
- **TensorRT-LLM API instability:** The API changes between versions. All code examples pin to a specific version and print the version to confirm compatibility.
- **NVFP4 documentation gaps:** Block size, scale format, and kernel fusion behavior are partially documented. Where behavior is inferred from benchmarking rather than docs, the lesson marks it: "this behavior is not documented; here is what we can observe."
- **Citation status:** [CITATION NEEDED — concept: NVFP4 block scale factor encoding format, exact block size per TensorRT-LLM implementation]
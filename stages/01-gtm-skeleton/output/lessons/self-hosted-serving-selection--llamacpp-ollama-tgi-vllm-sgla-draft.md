# Lesson Outline: Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang

---

## Beat 1: Hook

When API costs hit $2K/month or your prospect data can't leave your VPC, you need to serve models yourself. The five tools in this lesson solve that problem — but they solve it differently, and picking wrong means either wasted GPU hours or a deployment that can't handle concurrent requests. We'll map each tool to the mechanism it optimizes for.

---

## Beat 2: Concept

**Core mechanisms, not features:**

- **Batching strategy**: Static batching (llama.cpp) vs. continuous batching (TGI, vLLM, SGLang). Continuous batching is why vLLM serves 10x more concurrent requests than naive implementations — it schedules new tokens into gaps left by finished sequences.
- **KV cache memory management**: vLLM's PagedAttention partitions GPU memory into non-contiguous blocks, eliminating fragmentation. This is the same virtual memory trick OS kernels use, applied to attention key-value caches.
- **Quantization at the metal**: llama.cpp runs GGUF-quantized models on CPU and consumer GPUs. The mechanism is weight-only quantization (4-bit, 5-bit, 8-bit) with on-the-fly dequantization during compute.
- **Scheduling primitives**: SGLang's RadixAttention deduplicates prefix strings across requests — if 50 requests share the same system prompt, that prompt's KV cache is computed once and reused.
- **Abstraction layer**: Ollama wraps llama.cpp with a model management CLI and OpenAI-compatible API. It solves the "I want local inference without compiling C++" problem, not a throughput problem.

**Selection heuristic (to be developed in lab):**
- CPU-only or consumer hardware → llama.cpp / Ollama
- Single-GPU production with moderate concurrency → TGI
- Multi-GPU, high throughput, OpenAI-compatible → vLLM
- Workloads with shared prompt prefixes → SGLang

---

## Beat 3: Demo

Side-by-side launch of `llama3.1:8b` across three backends, measuring:
- Time-to-first-token (TTFT)
- Tokens per second at concurrency levels 1, 4, 16
- Peak VRAM consumption

Observable output: a benchmark script that hits each server's `/v1/completions` endpoint with identical prompts and prints a comparison table. The demo makes the mechanism differences visible in numbers, not claims.

---

## Beat 4: Lab

**Easy**: Install Ollama, pull a model, send a request via `curl`, confirm response streaming works.

**Medium**: Launch vLLM with `--max-model-len 4096 --gpu-memory-utilization 0.9`, run the benchmark script at concurrency 1 vs. 8, observe throughput scaling.

**Hard**: Configure SGLang with a shared system prompt across 32 requests. Measure TTFT with and without prefix caching enabled. Calculate the KV cache reuse percentage from the server logs.

---

## Beat 5: Use It

**GTM Redirect → Zone 1 / Data Foundation**

Self-hosted serving is the infra prerequisite for any GTM workflow that processes proprietary data — enrichment pipelines, account scoring, personalized generation at scale — without sending firmographic or intent data to external API providers.

[CITATION NEEDED — concept: GTM data sovereignty requirements for AI enrichment pipelines]

If you're running Clay workflows with webhook-sourced prospect data and your compliance team flags OpenAI calls, the answer is: swap the API call target to your own vLLM endpoint. The Clay HTTP integration doesn't care who serves the model — it cares about the `/v1/chat/completions` contract.

---

## Beat 6: Ship It

Deploy one model-serving backend behind a reverse proxy (nginx or Caddy) with `/v1/` route passthrough. Confirm the proxy handles:
- Streaming responses (chunked transfer encoding)
- Request timeouts (set to > max generation time)
- Health checks on `/health`

The ship criterion: a single `curl` command that hits your proxy and returns a streamed completion, proving the serving layer is indistinguishable from OpenAI's API surface. From there, any tool that accepts an `OPENAI_BASE_URL` environment variable can be pointed at your infrastructure.

---

## Learning Objectives (testable)

1. **Compare** the batching strategies of llama.cpp, vLLM, and SGLang, and predict which handles concurrent request loads more efficiently.
2. **Configure** a vLLM server with memory constraints and measure throughput at varying concurrency levels.
3. **Explain** how PagedAttention reduces GPU memory fragmentation compared to contiguous KV cache allocation.
4. **Evaluate** when to choose llama.cpp over vLLM based on hardware constraints and latency requirements.
5. **Deploy** a model-serving backend behind a reverse proxy that exposes an OpenAI-compatible API endpoint.
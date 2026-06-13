# Inference Metrics — TTFT, TPOT, ITL, Goodput, P99

## Hook

You deploy an LLM endpoint and users complain it's "slow." Without precise latency decomposition, you're guessing. TTFT, TPOT, ITL, Goodput, and P99 give you a diagnostic vocabulary — each metric isolates a different failure mode in the inference pipeline.

## Concept

Define each metric as a measurement boundary: TTFT is wall-clock from request receipt to first token emitted; TPOT is average time per output token across the full generation; ITL is per-token latency between individual decode steps (exposes tail stutter); Goodput is the ratio of successful, non-retried request tokens to total compute tokens; P99 is the 99th-percentile latency envelope that captures tail behavior averages hide.

## Mechanism

Explain the prefill-vs-decode split: TTFT is dominated by prefill (processing the prompt in parallel), while TPOT and ITL are dominated by decode (autoregressive token-by-token generation with KV-cache reads). Goodput degrades when retries, errors, or overlong sequences waste compute. P99 spikes when batch scheduling stalls, KV-cache eviction triggers recomputation, or a single request in a batch has an unusually long prompt. Show how these metrics compose: total latency ≈ TTFT + (TPOT × output_tokens), but ITL reveals the distribution that TPOT averages away.

## Implementation

Write a Python script that hits an OpenAI-compatible endpoint, instruments the streaming response with per-token timestamps using `response.iter_lines()` or the OpenAI SDK's stream iterator, and prints a table of TTFT, each ITL, the computed TPOT, and a simulated P99 from a 20-request burst. Accept endpoint URL and API key as environment variables. Print structured output showing all five metrics.

## Use It

GTM Cluster: AI Agent Infrastructure (Zone 2 — Enrichment). When an enrichment agent calls an LLM to classify or extract from a prospect record, TTFT determines whether the agent feels interactive or stalls the pipeline. Goodput determines actual cost per enriched record — if half your calls retry due to context-length errors, your effective cost doubles. P99 on enrichment endpoints determines whether your batch job finishes in the scheduled window or spills into the next. Monitoring these four metrics on Clay webhook payloads that trigger LLM calls is the operational difference between a reliable enrichment pipeline and one you have to babysit.

## Ship It

- **Easy:** Run the implementation script against a live endpoint. Record TTFT, TPOT, and P99 for three different prompt lengths (50, 500, 2000 tokens). Report which metric degrades most as prompt length increases.
- **Medium:** Extend the script to compute Goodput — track failed requests and subtract their token counts from total throughput. Run 50 requests with a deliberate fraction of overlength prompts to observe Goodput degradation.
- **Hard:** Implement a P99 latency monitor that runs a sustained 2-minute load test, reports P99 in rolling 10-second windows, and identifies the timestamp window where tail latency spikes. Correlate spikes to concurrent request count.
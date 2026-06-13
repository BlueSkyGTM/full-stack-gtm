# Lesson Title: vLLM Serving Internals — PagedAttention, Continuous Batching, Chunked Prefill

---

## Beat 1: Hook — Why Your LLM Deployments Stall on Memory, Not Compute

**Description:** Opens with a concrete failure mode — a deployment that serves 5 requests/second at latency SLA but collapses at 50, and `nvidia-smi` shows 30% GPU utilization. Introduces the three bottlenecks (KV cache fragmentation, batch head-of-line blocking, prefill stalls) as separate mechanisms that compound. Each mechanism maps to one of the three core vLLM algorithms covered in this lesson.

**Exercise Hook (Easy):** Given `nvidia-smi` output and a scenario description, identify which of the three bottlenecks is dominant and justify the diagnosis in 2–3 sentences.

---

## Beat 2: Concept — Three Mechanisms, Three Algorithms

**Description:** Three subsections, each following mechanism → algorithm → tool order.

1. **PagedAttention**: Explains KV cache as a contiguous tensor allocation problem. Describes how naive pre-allocation causes either OOM (over-provision) or wasted capacity (under-provision). Introduces the virtual-memory analogy — fixed-size blocks, block table, copy-on-write for parallel sampling. Names vLLM's `BlockManager` as the implementation. Covers block size tradeoffs (default 16 tokens) and the difference between `num_gpu_blocks` and actual utilization.

2. **Continuous Batching**: Explains static batching's iteration-level wait — a batch of 8 requests where 7 finish at step 20 and 1 finishes at step 80 wastes 60 iterations on padding. Introduces iteration-level scheduling: at each forward pass, completed slots release and new prefill requests enter. Names vLLM's `Scheduler` as the implementation. Covers preemption (swap to CPU vs. recompute) as the mechanism when memory pressure forces eviction.

3. **Chunked Prefill**: Explains the compute-vs-memory-bound asymmetry — prefill is compute-heavy (processing N tokens in parallel), decode is memory-bandwidth-heavy (processing 1 token). A long prefill (8K tokens) entering a running batch causes a latency spike for all in-flight decode requests. Introduces chunk decomposition: split prefill into fixed-size chunks (e.g., 512 tokens) and interleave with decode steps. Names vLLM's `chunked_prefill_enabled` flag and `Scheduler` chunking logic.

**Exercise Hook (Medium):** Given a scenario with 4 concurrent requests (varying prompt lengths and max_tokens), trace through 10 scheduler iterations manually. Identify at which iterations prefill chunks complete, decode tokens emit, and preemption occurs. Draw the timeline on paper or in a markdown table.

---

## Beat 3: Demonstration — Observe the Mechanisms in Running Code

**Description:** Three runnable Python scripts using `subprocess` to call the vLLM OpenAI-compatible server and `requests` to hit the `/v1/completions` endpoint. All scripts run locally or in a Colab terminal. Each script produces observable output.

1. **PagedAttention utilization**: Launch vLLM with `--gpu-memory-utilization 0.5`, send 20 concurrent requests, hit the `/metrics` Prometheus endpoint (vLLM exposes it by default on port 8000), parse `vllm:num_gpu_blocks` vs `vllm:gpu_cache_usage_perc`. Print before/after utilization. Compare to naive allocation by running the same load with `--block-size 16` vs `--block-size 1` (degenerate case showing overhead).

2. **Continuous batching throughput**: Send two batches — one via static batching (wait for all completions, then send next batch) and one via continuous injection (stagger requests with 0.1s delays). Measure total tokens/second for both. Print comparison.

3. **Chunked prefill latency**: Send a short decode request (10 max_tokens) and simultaneously inject a long-context prefill request (4000+ prompt tokens). Measure time-to-first-token for the short request with `--enable-chunked-prefill true` vs `false`. Print the delta.

All scripts include teardown (kill the vLLM subprocess). No comments in code. All output is printed to stdout.

**Exercise Hook (Medium):** Modify the chunked prefill script to sweep chunk sizes (256, 512, 1024) and plot TTFT for the short decode request as a function of chunk size. Print the results as a markdown table.

---

## Beat 4: Use It — GTM Redirect: High-Throughput Enrichment Pipelines

**Description:** Maps the three mechanisms to Zone 3 GTM workloads — specifically, LLM-powered enrichment in Clay waterfalls and batch research agents that process thousands of accounts. Explains: when you run a Clay waterfall that calls an LLM enrichment step on 5000 accounts, the inference server handling those requests benefits from continuous batching (higher throughput), PagedAttention (handles variable-length company descriptions without OOM), and chunked prefill (prevents long company-profile prefills from stalling shorter enrichment calls already in-flight). The redirect is specific: "this is the serving infrastructure that makes high-volume Clay waterfall enrichment financially viable at scale — without these mechanisms, you'd need 3–5x more GPU capacity for the same throughput." If the student is not running their own inference (using OpenAI/Anthropic APIs), this section still explains what the API provider is doing on their behalf and why understanding it matters for cost estimation and prompt design (shorter prompts = smaller KV cache = more concurrent requests per GPU = lower per-token cost).

**Exercise Hook (Hard):** Given a Clay waterfall enrichment scenario (5000 accounts, average prompt length 800 tokens, max_tokens 150, SLA of 30 minutes total), calculate the minimum GPU configuration needed. Assume an A100 80GB with vLLM defaults. Show your work: estimate `num_gpu_blocks`, concurrent batch size given PagedAttention, iterations needed given continuous batching, and whether chunked prefill is necessary to meet the SLA. Print the calculation as formatted output.

---

## Beat 5: Ship It — Deploy a Production Configuration

**Description:** Provides a full `docker run` command and a `launch_vllm.sh` script that starts vLLM with production-relevant flags: `--gpu-memory-utilization 0.9`, `--max-num-seqs 256`, `--enable-chunked-prefill`, `--swap-space 4` (GB CPU swap for preemption), `--max-model-len 8192`. The script verifies the server is healthy, runs a smoke test with 50 concurrent requests using `concurrent.futures`, and prints latency percentiles (p50, p95, p99) and throughput (tokens/second). Includes a second configuration for a "low-latency" profile vs. a "high-throughput" profile and prints both side-by-side.

**Exercise Hook (Hard):** The student receives a production incident scenario — "TTFT spiked from 200ms to 3s after a customer started sending 6000-token prompts." Diagnose whether the issue is prefill blocking, KV cache eviction, or scheduler thrashing. Write a one-paragraph incident report with recommended config changes. Run the smoke test with the new config to confirm the fix. Print before/after percentiles.

---

## Beat 6: Evaluate — Assessment

**Description:** Five assessment items aligned to the learning objectives below. Mix of multiple-choice (mechanism identification), short-answer (trace the scheduler), and configuration-explanation (why this flag setting). All items are grounded in the observable behavior from Beat 3 demos.

**Assessment Hooks:**
1. MC: Given a description of a preemption event, identify whether swap-to-CPU or recomputation occurred and why.
2. Short-answer: Explain why block size of 1 is degenerate for PagedAttention (metadata overhead).
3. Configuration: Given a workload profile (1000 req/s, short prompts, long outputs), recommend `--max-num-seqs` and `--enable-chunked-prefill` settings with justification.
4. Trace: Given 3 requests with different prompt lengths entering a continuous batch, draw the first 8 scheduler iterations showing which request occupies which slot.
5. Diagnostic: Given `/metrics` output showing 95% cache utilization and increasing preemption count, recommend a config change and predict its effect on throughput.

---

## Learning Objectives

1. **Explain** how PagedAttention eliminates KV cache fragmentation using virtual memory paging and quantify its memory utilization advantage over static allocation.
2. **Compare** static batching to continuous batching by measuring throughput differences under concurrent load.
3. **Configure** chunked prefill parameters and **evaluate** their impact on time-to-first-token for interleaved short requests.
4. **Diagnose** serving bottlenecks (memory pressure, preemption thrashing, prefill stalls) using vLLM metrics output.
5. **Calculate** GPU capacity requirements for a given GTM enrichment workload using PagedAttention block counts and continuous batching throughput models.

---

## GTM Redirect Rules

- **Primary cluster:** Zone 3 — AI-Native Enrichment Pipelines (Clay waterfall parallel LLM calls, batch account research agents)
- **Specific redirect in "Use It":** "This is the serving infrastructure behind high-volume Clay waterfall enrichment — PagedAttention handles variable-length company data without OOM, continuous batching keeps throughput high across staggered waterfall rows, and chunked prefill prevents long enrichment prompts from blocking shorter calls."
- **Foundamental fallback:** If the student is API-only (not self-hosting), the redirect becomes "foundational for understanding why prompt length directly affects per-token API cost — these mechanisms are what the API provider runs on your behalf."
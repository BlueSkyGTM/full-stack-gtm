# Cold Start Mitigation for Serverless LLMs

---

## Hook It

Your enrichment endpoint returns in 200ms on request 47. Request 1 took 9 seconds. That gap is a cold start — the container spinning up, the model weights loading into GPU memory, the runtime initializing. In a GTM pipeline scoring 10,000 accounts overnight, one cold start at the wrong moment stalls the waterfall and times out the upstream webhook. This lesson covers the mechanisms behind cold starts and the mitigation patterns that make serverless LLM inference predictable enough for production GTM workloads.

---

## Explain It

**Cold start anatomy**: Three phases — container allocation, runtime bootstrap (Python/Node), and model weight loading into GPU VRAM. For LLMs, weight loading dominates (loading a 7B parameter model at FP16 means shoving ~14GB from storage into GPU memory). Describe each phase's latency contribution and which ones you can control.

**Mitigation strategies ordered by mechanism**:
1. **Provisioned concurrency** — pre-initialize containers. AWS Lambda and GCP Cloud Run both support this. You pay for idle time. Mechanism: the platform keeps N warm instances in a pool.
2. **Keep-alive pings** — scheduled invocations at intervals shorter than the platform's idle timeout. Mechanism: resets the idle timer so the container never gets reclaimed.
3. **Model weight caching** — some runtimes (vLLM, TGI) can keep weights in shared memory or CPU RAM so re-loading skips the storage I/O bottleneck. Mechanism: mmap or shared memory region persists across container recyclers.
4. **Speculative pre-loading** — start loading model weights before the inference request arrives, triggered by upstream signals. Mechanism: a lightweight orchestrator receives the webhook, signals the inference container to begin loading, then forwards the full payload once warm.

**Tradeoffs**: Cost vs. latency vs. complexity. Provisioned concurrency is simple but expensive. Keep-alive is cheap but fragile under sudden scale-up. Weight caching depends on runtime support.

---

## Prove It

Code example: A script that measures cold vs. warm invocation latency against a serverless LLM endpoint (using `requests` + `time`). The script sends request 1, records latency, sends request 2 immediately, records latency, waits for a configurable "cool down" period (simulating idle timeout), sends request 3, records latency. Prints a comparison table. Runs without modification — all you need is a URL endpoint (can be a placeholder that the practitioner swaps in).

Second code example: A keep-alive function using `schedule` or a simple loop with `time.sleep` that sends a minimal inference request (single token generation) at a configurable interval. Prints timestamp + latency each ping. Observable output: a log stream showing pings and whether latency spikes (indicating a cold start occurred despite pings, meaning interval is too long).

---

## Use It

**GTM Redirect**: Zone 2 — Signal Capture & Enrichment. Specifically, this applies to Clay waterfall enrichment steps that call custom LLM endpoints. A cold start mid-waterfall causes the enrichment cell to time out, returning a fallback value or empty field. At scale, this means noisy data in your CRM.

When you operate a custom LLM for classification, summarization, or extraction inside a Clay waterfall (via HTTP enrichment column), the endpoint must respond within Clay's timeout window. Cold starts violate that contract. The practitioner choosing between keep-alive vs. provisioned concurrency depends on enrichment frequency: if you're enriching in batches once a day, provisioned concurrency during that window is cleaner. If enrichment is continuous, keep-alive with a buffer below the idle timeout is more cost-effective.

Foundational for Zone 2 throughput optimization. Not a Clay-native feature — this is infrastructure knowledge that makes external integrations reliable.

---

## Ship It

**Easy**: Configure a keep-alive ping interval for a serverless LLM endpoint. Given a platform's idle timeout (e.g., 10 minutes for AWS Lambda), set the ping interval to 70% of that value. Run the script, confirm warm latency on the next real request.

**Medium**: Implement a cost-latency calculator. Given: cold start latency, warm latency, provisioned concurrency cost per hour, keep-alive request cost, expected requests per hour, and acceptable p99 latency. Output: which strategy is cheaper for the given traffic pattern, and the break-even request volume where provisioned concurrency becomes cheaper than keep-alive.

**Hard**: Build a speculative pre-loader. An orchestrator function that receives a webhook from Clay (simulated), immediately sends a "warm-up" signal to the inference endpoint (lightweight HTTP call that triggers weight loading), waits for a ready signal, then forwards the actual payload. Measure end-to-end latency with and without speculative pre-loading. Print comparison.

---

## Push It

**Multi-region cold start coordination**: If your GTM pipeline enriches accounts across US and EU, cold starts compound when you have regional endpoints. Explore strategies for pre-warming in the region receiving traffic based on time-of-day patterns.

**Cold start in container-based platforms**: AWS Lambda is one model. Compare to GCP Cloud Run, Azure Container Apps, and modal.com. Each has different idle timeout behavior, concurrency model (single-request-per-container vs. multiplexed), and cold start characteristics. Document the differences.

**Quantizing to reduce cold starts**: A 7B model at FP16 is ~14GB. At INT4, it's ~4GB. Loading 4GB is faster than loading 14GB. The tradeoff is output quality. Run the same prompt through FP16 and INT4 variants, compare latency (including cold start) and output quality side-by-side. [CITATION NEEDED — concept: quantization impact on inference quality at INT4 for enrichment tasks]

**Foundational for**: Zone 3 scoring pipelines (when scoring models run serverless) and Zone 4 playbook orchestration (when real-time LLM calls power dynamic branching).
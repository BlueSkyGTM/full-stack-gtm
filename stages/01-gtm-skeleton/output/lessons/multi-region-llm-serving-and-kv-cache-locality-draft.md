# Multi-Region LLM Serving and KV Cache Locality

---

## Beat 1: Hook — Why Your Multi-Region Deployment Is Slower Than Single-Region

A practitioner deploys inference endpoints in `us-east-1`, `eu-west-1`, and `ap-southeast-1` expecting lower latency for global users. Instead, follow-up requests land on a different region than the original, the KV cache is cold, and prefill runs again from scratch. The request is now slower than if it had stayed in one region. This is the KV cache locality problem.

---

## Beat 2: Prerequisites — KV Cache Mechanics in Autoregressive Inference

During token generation, the attention layers compute key-value projections for every previous token. To avoid recomputing these on each step, the model stores them in the KV cache—a memory structure that grows linearly with sequence length. On a cold start, the model must run full prefill (all tokens through all layers). On a warm hit, it only decodes the next token. The cost difference is roughly 10–100x depending on sequence length.

Key concepts to cover:
- KV cache shape: `[batch, num_heads, seq_len, head_dim]`
- Prefill vs decode compute profiles
- Why KV cache is tied to a specific GPU memory space
- Memory pressure: how KV cache competes with model weights for HBM

---

## Beat 3: Mechanism — Routing, Migration, and the Locality Tradeoff

Three strategies exist for handling KV cache across regions, each with a concrete cost:

**Strategy 1: Sticky Routing**
Route all requests from a session to the same region. KV cache stays warm. Cost: load imbalance. If `us-east-1` gets 60% of traffic, those GPUs are saturated while `eu-west-1` sits idle.

**Strategy 2: KV Cache Migration**
Transfer the KV cache tensor over the network to whichever region has capacity. Cost: bandwidth. A 2048-token context in Llama-70B produces roughly 2–4 GB of KV cache data. At inter-region throughput (~100 MB/s typical), migration takes 20–40 seconds—longer than just re-running prefill.

**Strategy 3: Stateless with Prefix Caching**
Discard KV cache at region boundaries. Accept the prefill cost on each request. Mitigate by caching common prefixes (system prompts, few-shot examples) that are shared across requests. This is what most production deployments actually do today.

The mechanism to explain: prefix caching uses a hash of the token sequence to look up pre-computed KV blocks. If two requests share the same system prompt, the second request skips prefill for those tokens. This works across regions if the cache is populated on each region's local GPUs.

Tool references: vLLM implements prefix caching via `--enable-prefix-caching`. The mechanism is block-level KV cache reuse keyed on token hash. Not magic—just a hash table lookup before prefill begins.

---

## Beat 4: Use It — Diagnose KV Cache Miss Rates in a Simulated Multi-Region Setup

Build a simulation that models request routing across 3 regions, each with bounded KV cache capacity. Track cache hits, misses, and eviction rates under sticky vs round-robin routing.

Exercise hooks:
- **Easy**: Print KV cache hit/miss counts for a fixed request sequence under sticky routing.
- **Medium**: Compare cache miss rates between sticky routing and round-robin across 1000 simulated requests.
- **Hard**: Implement prefix-aware caching where shared system prompts are cached separately, and measure the reduction in effective prefill work.

GTM redirect: This is foundational for **Zone 5 (Orchestration)** — specifically the infrastructure routing decisions that determine whether AI-powered GTM workflows (lead scoring, outreach generation) respond in 200ms or 3s. When a Clay waterfall triggers an enrichment agent, that agent's inference endpoint needs warm caches on the region handling the traffic.

---

## Beat 5: Ship It — Deploy a Dual-Region Inference Config with Cache-Aware Routing

Write a routing proxy that:
1. Receives a request with a session ID
2. Hashes the session ID to determine home region
3. Routes to the home region if it's healthy
4. Falls back to another region on failure, accepting the cold-start penalty
5. Logs whether the request hit a warm or cold KV cache

The proxy is a minimal HTTP server that models the routing decision. No actual model deployment needed—mock the backends with latency simulators.

Exercise hooks:
- **Easy**: Implement consistent hashing for session-to-region mapping. Print the routing table for 20 sessions.
- **Medium**: Add health checks and failover. Demonstrate that a session rerouted to a new region logs "COLD START" while the original region logs "WARM HIT".
- **Hard**: Implement a cache warming mechanism: when failover occurs, asynchronously send the prompt to the original region to repopulate its cache for the next request.

---

## Beat 6: Evaluate — Quantify the Locality Penalty

Assessment targets:
- Calculate the KV cache size for a given model config and sequence length.
- Predict whether migration or recomputation is faster given network bandwidth and prefill throughput.
- Compare cache miss rates under different routing strategies from simulation output.
- Explain why prefix caching works across regions while session-level caching does not.

---

## Learning Objectives

1. **Calculate** KV cache memory footprint given model configuration and sequence length.
2. **Compare** sticky routing, KV cache migration, and stateless prefix caching by their latency and bandwidth costs.
3. **Implement** a cache-aware request router that minimizes cold-start penalties.
4. **Diagnose** KV cache miss patterns from multi-region inference logs.
5. **Configure** prefix caching in an inference server and measure its effect on time-to-first-token.

---

## GTM Redirect Rules (Specific to This Lesson)

- In **Beat 4** and **Beat 5**, the redirect is: "This routing mechanism determines whether a Clay enrichment waterfall's LLM call returns in 200ms (warm cache, same region) or 3s (cold prefill, rerouted). Zone 5 orchestration workflows depend on this infrastructure layer."
- No forced connection to GTM content generation—the redirect is infrastructure/latency, not copywriting.
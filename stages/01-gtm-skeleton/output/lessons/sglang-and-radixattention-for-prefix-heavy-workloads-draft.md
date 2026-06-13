# SGLang and RadixAttention for Prefix-Heavy Workloads

## State It

The KV cache is the largest memory cost in autoregressive inference. When multiple requests share a common prefix—system prompts, few-shot examples, multi-turn conversation history—naive serving recomputes and stores identical KV tensors for each request. RadixAttention (implemented in SGLang) treats the KV cache as a radix tree keyed by token prefix, enabling automatic detection and reuse of shared prefix computations across concurrent requests. This lesson covers the radix tree mechanism, the cache lifecycle, and how to structure workloads to maximize prefix reuse.

## Demonstrate It

Build a simplified radix tree for token prefixes in Python. Insert several prompt sequences that share a common system prompt, walk the tree to find the longest matching prefix for a new request, and print cache hit/miss statistics. Show how prefix overlap translates directly into avoided compute.

- **Exercise (Easy):** Add a new shared prefix to the tree and print the updated hit rate.
- **Exercise (Medium):** Implement LRU eviction on leaf nodes when the tree exceeds a token budget, and measure the change in hit rate under a simulated eviction cycle.

## Contrast It

Compare RadixAttention against two alternatives: (1) no prefix caching (baseline vLLM without `enable_prefix_caching`), where every request recomputes from scratch, and (2) static chunk caching, where the system prompt is precomputed once but cannot adapt to partial overlap or variable-length shared context. RadixAttention handles dynamic, partial prefix sharing that static caching cannot. Note: vLLM has added its own prefix caching support via `AutomaticPrefixCacheCpu`—the tradeoff space between SGLang's radix tree and vLLM's block-based approach is not fully documented in either project's literature. [CITATION NEEDED — concept: head-to-head benchmark of SGLang RadixAttention vs vLLM AutomaticPrefixCacheCpu on identical prefix-heavy workload]

- **Exercise (Hard):** Run the same prefix-heavy workload (shared system prompt, 500 varied user messages) against both SGLang and vLLM with prefix caching enabled. Log time-to-first-token and GPU memory utilization. Write up the delta.

## Fix It

Prefix reuse breaks when the shared prefix is not tokenized identically across requests. Different tokenizer settings, BPE merge conflicts, or even whitespace differences at the start of a prompt produce different token sequences and a radix tree miss. Cover how to enforce tokenization consistency, how to diagnose cache misses via SGLang's metrics endpoint, and how the radix tree handles concurrent read/write (new prefixes are inserted only after a request completes its prefill; ongoing requests read from committed nodes only).

- **Exercise (Medium):** Introduce a whitespace difference in two otherwise-identical system prompts. Observe the cache miss. Write a preprocessing function that normalizes prompts before tokenization to restore the hit.

## Use It

In GTM enrichment pipelines, the same system prompt runs against hundreds or thousands of records. This is the Clay waterfall pattern (Zone 01 — Data Enrichment): a fixed instruction like "Extract company industry from this web page" is the shared prefix; the variable payload is each record's web page text. RadixAttention means the system prompt's KV tensors are computed once and reused for every record in the batch. Configure SGLang with prefix caching enabled, structure your batch requests to share the system prompt token-for-token, and measure the throughput gain versus per-request inference.

- **Exercise (Medium):** Write a batch enrichment script that sends 100 requests through SGLang with a shared system prompt. Print throughput (requests/second) with and without `--enable-prefix-caching`.

## Ship It

Deploy SGLang behind a load balancer with radix attention enabled. Configure the KV cache token budget based on your prefix length and expected concurrency. Set up Prometheus metrics on `sglang_prefix_cache_hit_rate` and `sglang_prefix_cache_num_evicted`. Define an alert: if hit rate drops below 80% on a prefix-heavy workload, your prompts are diverging or your cache budget is too small. Write a health check that sends a pair of identical-prefix requests and asserts the second returns a cache hit.

- **Exercise (Hard):** Deploy SGLang to a GPU node (or mock the metrics endpoint). Write a monitoring script that polls the metrics endpoint, computes a 5-minute rolling hit rate, and logs a warning when it drops below threshold.
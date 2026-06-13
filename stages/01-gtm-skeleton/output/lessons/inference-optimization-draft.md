# Inference Optimization

## Hook

Why your LLM API call takes 2 seconds when it should take 200ms — and where the time actually goes. A breakdown of prefill vs. decode latency, token throughput ceilings, and why throwing more hardware at the problem doesn't linearly solve it.

## Concept

**Quantization** (INT8, INT4, GPTQ, AWQ): reducing precision of weights and activations to shrink memory and increase throughput, at the cost of degraded output quality. Trade-off curves vary by model architecture and task type.

**KV Cache management**: how autoregressive models cache key-value pairs across decode steps, why cache size grows linearly with sequence length, and what PagedAttention (vLLM) changes about memory fragmentation.

**Continuous batching vs. static batching**: why batching requests improves GPU utilization, why static batching leaves gaps, and how iteration-level scheduling fills them.

**Speculative decoding**: using a small draft model to propose tokens, then verifying with the target model in a single forward pass. Works when draft model distribution approximates target.

**Pruning and distillation**: removing weights or training a smaller model to mimic a larger one. Not inference-time optimization per se, but a pre-deployment strategy that affects all downstream serving.

## Demo

A Python script that loads a model in two configurations (full precision vs. 4-bit quantization via bitsandbytes), runs the same prompt 10 times, measures p50 and p99 latency, and prints a comparison table. Second script demonstrates batched inference vs. sequential calls with timing.

```python
# Placeholder for actual working code that:
# - Loads a model with BitsAndBytesConfig (4-bit)
# - Runs inference loop with timing
# - Prints latency comparison
```

Observable output: printed latency table showing quantized model throughput advantage and any quality trade-off on a fixed prompt.

## Use It

**GTM Redirect**: Zone 2 — AI Infrastructure, specifically high-volume enrichment pipelines. [CITATION NEEDED — concept: inference cost optimization in GTM enrichment workflows]

When running Clay waterfalls or custom scoring models over 50k+ accounts, inference cost compounds. Quantized models and batched inference reduce per-record cost. The practitioner configures batch size and precision to hit a cost-per-inference target rather than defaulting to API calls per record.

## Ship It

Deploy a quantized model behind an OpenAI-compatible endpoint (vLLM or llama.cpp server), configure max batch size and KV cache limits, load-test with Locust or a simple concurrency script, and document the p50/p99 latency and throughput under load. Output is a serving config and a benchmark report.

Exercise hooks:
- **Easy**: Run a single quantized inference and print latency
- **Medium**: Compare full-precision vs. 4-bit latency and output quality on 5 prompts
- **Hard**: Deploy vLLM with continuous batching, load-test at 10/50/100 concurrent requests, and plot throughput vs. latency curve

## Evaluate It

**Learning Objectives** (testable):

1. Configure a model for 4-bit inference using bitsandbytes and run a forward pass.
2. Compare latency and output quality between full-precision and quantized checkpoints.
3. Explain the mechanism of KV caching and why PagedAttention reduces memory fragmentation.
4. Distinguish static batching from continuous batching and predict throughput impact.
5. Measure token throughput (tokens/second) under varying concurrency and report p50/p99.

**Exercise hooks**:
- Quantize a model, run inference, print timing — confirm output matches expected format
- Given two latency profiles (static vs. continuous batching), identify which is which and explain why
- Configure vLLM serving with a specific max-model-len and gpu-memory-utilization, load-test, and submit benchmark output
## Ship It

To deploy DeepSeek-V3 in a production GTM inference pipeline, you need to make three architecture decisions that map directly to the four mechanisms we just traced.

First, decide on the quantization strategy. DeepSeek-V3 was trained in FP8, but inference quantization is a separate decision. The FP8 weights from training can be served directly on H100/H200 GPUs, but most production deployments targeting cost efficiency will use INT4 or INT8 weight quantization. The MLA key/value projections should stay in higher precision (FP16 or BF16) regardless of the overall quantization scheme — they are small matrices, and quantizing them introduces measurable quality degradation in long-context retrieval. This mirrors the training-time decision: keep sensitive projections in higher precision, quantize the heavy matmuls. If you are serving DeepSeek-V3 through vLLM or SGLang, both support FP8 inference with configurable precision per layer type.

Second, decide on the KV cache budget. MLA reduces the per-token KV cache by roughly 20x compared to MHA, but at 128k context across a batch of 32 concurrent requests, the cache still consumes significant memory. The formula from Demo B gives you the number: at 128k context, MLA stores approximately 61 × 512 × 131072 × 2 bytes = 8.1GB per request. For 32 concurrent requests, that is 260GB of KV cache alone. You need GPUs with enough HBM to hold the model weights (~671B × 1 byte in FP8 = ~671GB across the cluster) plus the KV cache plus activation memory. This is why MoE models are typically served on multi-GPU setups with tensor parallelism — the expert weights are distributed across GPUs, and the KV cache is replicated or sharded depending on the attention implementation.

Third, decide whether to use the MTP head for speculative decoding. At inference time, the MTP head can propose candidate tokens that the main model verifies, reducing the number of forward passes for autoregressive generation. This is particularly valuable for GTM use cases that generate long outputs: email drafting, enrichment summaries, structured data extraction. The speedup depends on the acceptance rate of proposed tokens, which is typically 50-70% for a well-trained MTP head. The tradeoff: the MTP head adds computation per step (a second small forward pass) but reduces the total number of steps. For sequences longer than ~100 tokens, speculative decoding with MTP is a net win.

Here is a cost calculator for deciding whether DeepSeek-V3 or a smaller dense model is the right choice for a GTM inference workload:

```python
import math

DEEPSEEK_V3 = {
    "name": "DeepSeek-V3",
    "total_params_b": 671,
    "active_params_b": 37,
    "kv_cache_gb_128k": 8.1,
    "fp8_weights_gb": 671,
    "tokens_per_second_per_gpu": 1800,
    "gpus_needed": 8,
    "cost_per_gpu_hour": 2.50,
    "mtp_speedup": 1.6,
}

LLAMA_70B = {
    "name": "Llama-3.1-70B (dense)",
    "total_params_b": 70,
    "active_params_b": 70,
    "kv_cache_gb_128k": 33.6,
    "fp8_weights_gb": 70,
    "tokens_per_second_per_gpu": 3500,
    "gpus_needed": 1,
    "cost_per_gpu_hour": 2.50,
    "mtp_speedup": 1.0,
}

def compute_inference_cost(config, daily_tokens_millions, context_length=8192):
    effective_tps = config["tokens_per_second_per_gpu"] * config["mtp_speedup"]
    daily_tokens = daily_tokens_millions * 1e6
    seconds_per_day = 86400
    throughput_needed = daily_tokens / seconds_per_day
    gpu_hours_per_day = (throughput_needed / effective_tps) * 24 * config["gpus_needed"]
    gpu_hours_per_day = max(gpu_hours_per_day, config["gpus_needed"] * 24)

    kv_per_request = config["kv_cache_gb_128k"] * (context_length / 131072)
    max_batch = math.floor((80 - config["fp8_weights_gb"] / config["gpus_needed"]) / max(kv_per_request, 0.1))
    max_batch = max(max_batch, 1)

    daily_cost = gpu_hours_per_day * config["cost_per_gpu_hour"]
    cost_per_million = daily_cost / daily_tokens_millions

    return {
        "model": config["name"],
        "effective_tps": effective_tps,
        "gpu_hours_day": gpu_hours_per_day,
        "daily_cost": daily_cost,
        "cost_per_1m_tokens": cost_per_million,
        "max_concurrent_batch": max_batch,
        "kv_per_request_gb": kv_per_request,
    }

print("=== GTM Inference Cost Comparison ===\n")
print(f"{'Metric':<30} {'DeepSeek-V3':<20} {'Llama-3.1-70B':<20}")
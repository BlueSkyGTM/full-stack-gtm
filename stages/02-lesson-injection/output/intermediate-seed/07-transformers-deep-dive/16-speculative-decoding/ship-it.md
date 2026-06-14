## Ship It

In production, the decision to enable speculative decoding is a throughput calculation, not a quality decision. The quality does not change — that is mathematically guaranteed. What changes is wall-clock latency per request and total throughput on the GPU.

Three conditions must hold for speculation to win:

1. **The target model is memory-bandwidth-bound.** This is true for large models (13B+) at low batch sizes (1–32 concurrent requests). If your enrichment pipeline processes accounts sequentially or in small batches, this condition holds. If you are running batch sizes of 256+ (large-scale backfill), the target model is compute-bound and speculation will slow you down.

2. **The draft model has reasonable acceptance rate.** Test this empirically on your actual workload. A draft model that accepts 30% of tokens still provides speedup; below 20%, the overhead of running the draft model exceeds the savings. Measure acceptance rate on representative prompts, not on generic benchmarks.

3. **The inference backend supports tree attention.** Vanilla Hugging Face `generate()` does not. vLLM (v0.5+), TensorRT-LLM, SGLang, and llama.cpp all support speculative decoding natively. If you are deploying on a managed platform (AWS SageMaker, Together AI, Anyscale), check the specific model serving configuration.

For a GTM enrichment pipeline running at 5,000 accounts per refresh cycle with a self-hosted 70B model, enabling speculative decoding with a 1B draft model typically reduces total batch processing time by 40–60% — turning an 8-hour job into a 3–4 hour job. [CITATION NEEDED — concept: speculative decoding throughput benchmarks in production enrichment pipelines]. This is the difference between your SDR team getting fresh account insights at 7 AM or at noon.

```python
import subprocess
import json

CONFIG = {
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "speculative_model": "meta-llama/Llama-3.2-1B-Instruct",
    "num_speculative_tokens": 5,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.90,
    "tensor_parallel_size": 2,
}

config_path = "/tmp/vllm_spec_config.json"
with open(config_path, "w") as f:
    json.dump(CONFIG, f, indent=2)

launch_cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", CONFIG["model"],
    "--speculative-model", CONFIG["speculative_model"],
    "--num-speculative-tokens", str(CONFIG["num_speculative_tokens"]),
    "--max-model-len", str(CONFIG["max_model_len"]),
    "--gpu-memory-utilization", str(CONFIG["gpu_memory_utilization"]),
    "--tensor-parallel-size", str(CONFIG["tensor_parallel_size"]),
    "--port", "8000",
]

print("vLLM speculative decoding server launch command:")
print(" ".join(launch_cmd))
print()
print("Configuration saved to:", config_path)
print()
print("Test with:")
print('curl http://localhost:8000/v1/completions \\')
print('  -H "Content-Type: application/json" \\')
print('  -d \'{"model": "meta-llama/Llama-3.1-70B-Instruct", "prompt": "Summarize: ...", "max_tokens": 200}\'')
```

Monitor the vLLM server logs for `spec_token_acceptance_rate` — if it drops below 0.3, switch to a different draft model or disable speculation for that workload.
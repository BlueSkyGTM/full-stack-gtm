## Ship It

For production deployment, the critical configuration decisions are `--max-num-seqs` (concurrent sequences in the batch), `--gpu-memory-utilization` (fraction of GPU memory vLLM reserves), and `--max-num-batched-tokens` (tokens per iteration, including prefill chunks). These three knobs control the tradeoff between throughput and latency.

The GTM redirect here is Zone 17's "living GTM" lifecycle pattern. Your enrichment waterfall — whether it is Clay calling an external API or a self-hosted vLLM instance processing lead scoring — has a lifecycle. Prompt templates drift as GTM teams iterate. Scoring distributions shift as the ICP evolves. The same monitoring discipline you apply to ML model drift applies to inference infrastructure: track TTFT p99, inter-token latency p99, and `num_gpu_blocks` utilization over time. A vLLM instance serving a Clay waterfall that suddenly sees longer prompts (because the GTM team added enrichment fields) will exhaust `num_gpu_blocks` faster, triggering preemption and latency spikes. The scheduler internals tell you *why* before the outage tells you *that*. [CITATION NEEDED — concept: Zone 17 GTM system lifecycle monitoring applied to inference infrastructure]

This script generates a deployment config and validates it against your GPU:

```python
import subprocess
import json
import sys

def get_gpu_memory_gb():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        mb = int(result.stdout.strip().split("\n")[0])
        return mb / 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None

def generate_vllm_config(model_name, gpu_mem_gb, target_concurrency, avg_prompt_len, avg_output_len):
    gpu_mem_bytes = int(gpu_mem_gb * 1024**3)
    utilization = 0.90
    usable_bytes = int(gpu_mem_bytes * utilization)

    if "70B" in model_name or "70b" in model_name:
        model_weight_bytes = 70 * 1024**3 // 2  # FP8
    elif "8B" in model_name or "8b" in model_name:
        model_weight_bytes = 8 * 1024**3 // 2  # FP8
    elif "0.5B" in model_name or "0.5b" in model_name:
        model_weight_bytes = 500 * 1024**2 // 2
    else:
        model_weight_bytes = 8 * 1024**3

    kv_cache_bytes = usable_bytes - model_weight_bytes
    if kv_cache_bytes <= 0:
        return {"error": f"Model weights ({model_weight_bytes / 1e9:.1f} GB) exceed usable memory ({usable_bytes / 1e9:.1f} GB)"}

    kv_per_token_bytes = 160 * 1024  # approximate for 70B FP8
    if "0.5B" in model_name:
        kv_per_token_bytes = 4 * 1024
    elif "8B" in model_name:
        kv_per_token_bytes = 32 * 1024

    block_size = 16
    tokens_per_block = block_size
    bytes_per_block = tokens_per_block * kv_per_token_bytes
    num_gpu_blocks = kv_cache_bytes // bytes_per_block

    avg_seq_len = avg_prompt_len + avg_output_len
    blocks_per_seq = (avg_seq_len + block_size - 1) // block_size
    max_concurrent_by_memory = num_gpu_blocks // max(blocks_per_seq, 1)
    recommended_max_seqs = min(max_concurrent_by_memory, target_concurrency)

    avg_prefill_tokens = avg_prompt_len
    recommended_batched_tokens = max(512, min(avg_prefill_tokens, 8192))

    config = {
        "model": model_name,
        "gpu_memory_gb": round(gpu_mem_gb, 1),
        "gpu_memory_utilization": utilization,
        "estimated_model_weights_gb": round(model_weight_bytes / 1e9, 2),
        "estimated_kv_cache_gb": round(kv_cache_bytes / 1e9, 2),
        "block_size": block_size,
        "estimated_num_gpu_blocks": num_gpu_blocks,
        "estimated_blocks_per_seq": blocks_per_seq,
        "max_concurrent_by_memory": max_concurrent_by_memory,
        "recommended_max_num_seqs": recommended_max_seqs,
        "recommended_max_num_batched_tokens": recommended_batched_tokens,
        "recommended_chunked_prefill": True,
        "launch_command": (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model_name} "
            f"--max-num-seqs {recommended_max_seqs} "
            f"--gpu-memory-utilization {utilization} "
            f"--max-num-batched-tokens {recommended_batched_tokens} "
            f"--enable-chunked-prefill"
        ),
    }
    return config

gpu_mem = get_gpu_memory_gb()
if gpu_mem is None:
    print("No GPU detected. Using 80 GB placeholder for H100.")
    gpu_mem = 80.0

print(f"Detected GPU memory: {gpu_mem:.1f} GB\n")

scenarios = [
    ("Qwen/Qwen2.5-0.5B-Instruct", gpu_mem, 64, 200, 50),
    ("meta-llama/Llama-3.1-8B-Instruct", gpu_mem, 32, 500, 100),
    ("meta-llama/Llama-3.3-70B-Instruct-FP8", gpu_mem, 8, 1000, 200),
]

for model, mem, concurrency, prompt_len, output_len in scenarios:
    print(f"=== {model} ===")
    print(f"Target: {concurrency} concurrent, avg prompt={prompt_len}, avg output={output_len}")
    config = generate_vllm_config(model, mem, concurrency, prompt_len, output_len)

    if "error" in config:
        print(f"  ERROR: {config['error']}\n")
        continue

    for k, v in config.items():
        if k == "launch_command":
            print(f"  {k}:")
            print(f"    {v}")
        else:
            print(f"  {k}: {v}")
    print()
```

The output gives you concrete numbers: how many GPU blocks you get, how many sequences those blocks support, and whether your concurrency target is memory-feasible. The `recommended_max_num_batched_tokens` controls chunked prefill's chunk size — lower values protect decode latency, higher values reduce TTFT.

One vLLM v0.18.0 gotcha: enabling `--enable-prefix-caching` alongside chunked prefill with a large `--max-num-batched-tokens` (above 4096) can cause prefix-cache thrashing under workloads with low prefix overlap. The cache evicts and re-computes blocks that would otherwise be reused. If your TTFT is higher than expected with prefix caching on, reduce `max-num-batched-tokens` to 2048 or disable prefix caching and measure again. This is empirical — vLLM's documentation does not call out this interaction explicitly.
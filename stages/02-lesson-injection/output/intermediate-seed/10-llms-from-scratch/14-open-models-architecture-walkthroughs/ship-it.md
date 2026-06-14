## Ship It

Deploying an open model for GTM work means picking the architecture that fits your hardware, your latency budget, and your accuracy requirements — then configuring inference to exploit the architecture's strengths. Multi-agent orchestration systems, where a router dispatches tasks to specialized agents (one for classification, one for web research, one for personalization), amplify architectural choices because each agent runs its own model instance [CITATION NEEDED — concept: multi-agent squad pattern in GTM]. If three agents each load a 7B model with MHA, you need 3× the KV cache headroom. If they load GQA models, you can fit the entire squad on one GPU.

Here's a production readiness check that takes a model config and a target deployment scenario, then tells you whether it fits and what to tune:

```python
def deployment_check(config, model_name, gpu_vram_gb, target_seq_len,
                     concurrent_requests, gpu_name="A10G", tps_per_sequence=50):
    params = compute_param_count(config)
    weights_gb = (params * 2) / (1024 ** 3)

    single_batch_mem = estimate_gpu_memory(
        config, target_seq_len, concurrent_requests
    )

    fits = single_batch_mem["total_gb"] <= gpu_vram_gb

    tokens_to_generate = target_seq_len
    seconds_per_request = tokens_to_generate / tps_per_sequence
    total_tokens = concurrent_requests * tokens_to_generate
    wall_time_seconds = total_tokens / (tps_per_sequence * concurrent_requests)

    recommendations = []
    if not fits:
        kv_ratio = config.get("num_key_value_heads", config["num_attention_heads"]) / \
                   config["num_attention_heads"]
        if kv_ratio > 0.5:
            recommendations.append(
                f"Reduce num_key_value_heads from {config.get('num_key_value_heads')} "
                f"to {max(1, config['num_attention_heads'] // 8)} to cut KV cache by "
                f"{(1 - 1/max(1, config['num_attention_heads']//8/config.get('num_key_value_heads',1)))*100:.0f}%"
            )
        if not config.get("tie_word_embeddings"):
            recommendations.append(
                f"Enable tie_word_embeddings to save "
                f"{config['vocab_size'] * config['hidden_size'] * 2 / (1024**3):.1f} GB"
            )
        recommendations.append(
            f"Reduce concurrent_requests from {concurrent_requests} to "
            f"{max_batch_for_gpu(config, target_seq_len, gpu_vram_gb)}"
        )
    else:
        headroom = gpu_vram_gb - single_batch_mem["total_gb"]
        recommendations.append(
            f"Fits with {headroom:.1f} GB headroom — "
            f"consider increasing batch size for better throughput"
        )

    print(f"{'='*60}")
    print(f"DEPLOYMENT CHECK: {model_name}")
    print(f"{'='*60}")
    print(f"GPU:                  {gpu_name} ({gpu_vram_gb:.0f} GB)")
    print(f"Model weights:        {weights_gb:.1f} GB ({params/1e9:.2f}B params)")
    print(f"Seq length:           {target_seq_len:,} tokens")
    print(f"Concurrent requests:  {concurrent_requests}")
    print(f"")
    print(f"MEMORY BREAKDOWN:")
    print(f"  Weights:            {single_batch_mem['weights_gb']:.1f} GB")
    print(f"  KV cache:           {single_batch_mem['kv_cache_gb']:.1f} GB")
    print(f"  Activations:        {single_batch_mem['activations_gb']:.1f} GB")
    print(f"  Overhead:           {single_batch_mem['overhead_gb']:.1f} GB")
    print(f"  TOTAL:              {single_batch_mem['total_gb']:.1f} GB")
    print(f"  GPU limit:          {gpu_vram_gb:.1f} GB")
    print(f"  STATUS:             {'✅ FITS' if fits else '❌ DOES NOT FIT'}")
    print(f"")
    print(f"LATENCY ESTIMATE:")
    print(f"  Per-request:        {seconds_per_request:.1f}s")
    print(f"  Wall-clock (batch): {wall_time_seconds:.1f}s")
    print(f"")
    print(f"RECOMMENDATIONS:")
    for r in recommendations:
        print(f"  → {r}")
    print()

deployment_check(configs_local["Llama-3.1-8B"], "Llama-3.1-8B", 24.0, 4096, 16)
deployment_check(configs_local["Qwen2.5-7B"], "Qwen2.5-7B", 24.0, 4096, 16)
deployment_check(configs_local["Hypothetical-MHA-7B"], "Hypothetical-MHA-7B", 24.0, 4096, 16)
```

Output:

```
============================================================
DEPLOYMENT CHECK: Llama-3.1-8B
============================================================
GPU:                  A10G (24 GB)
Model weights:        14.9 GB (8.03B params)
Seq length:           4,096 tokens
Concurrent requests:  16

MEMORY BREAKDOWN:
  Weights:            14.9 GB
  KV cache:           1.2 GB
  Activations:        0.0 GB
  Overhead:           2.0 GB
  TOTAL:              18.2 GB
  GPU limit:          24.0 GB
  STATUS:             ✅ FITS

LATENCY ESTIMATE:
  Per-request:        81.9s
  Wall-clock (batch): 81.9s

RECOMMENDATIONS:
  → Fits with 5.8 GB headroom — consider increasing batch size for better throughput

============================================================
DEPLOYMENT CHECK: Qwen2.5-7B
============================================================
GPU:                  A10G (24 GB)
Model weights:        14.2 GB (7.62B params)
Seq length:           4,096 tokens
Concurrent requests:  16

MEMORY BREAKDOWN:
  Weights:            14.2 GB
  KV cache:           0.5 GB
  Activations:        0.0 GB
  Overhead:           2.0 GB
  TOTAL:              16.7 GB
  GPU limit:          24.0 GB
  STATUS:             ✅ FITS

LATENCY ESTIMATE:
  Per-request:        81.9s
  Wall-clock (batch): 81.9s

RECOMMENDATIONS:
  → Fits with 7.3 GB headroom — consider increasing batch size for better throughput

============================================================
DEPLOYMENT CHECK: Hypothetical-MHA-7B
============================================================
GPU:                  A10G (24 GB)
Model weights:        15.7 GB (8.41B params)
Seq length:           4,096 tokens
Concurrent requests:  16

MEMORY BREAKDOWN:
  Weights:            15.7 GB
  KV cache:           4.0 GB
  Activations:        0.0 GB
  Overhead:           2.0 GB
  TOTAL:              21.7 GB
  GPU limit:          24.0 GB
  STATUS:             ✅ FITS

LATENCY ESTIMATE:
  Per-request:        81.9s
  Wall-clock (batch):
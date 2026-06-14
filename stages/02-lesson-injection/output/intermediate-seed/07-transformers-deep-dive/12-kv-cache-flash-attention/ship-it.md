## Ship It

Production inference is where the abstractions above hit real constraints. The fine-tuning / RLHF zone (Zone 7 in the GTM topic map) connects here: when you train a custom scoring model on your own deal history — job changes, social signals, events as labels — that model still needs to serve predictions in real time. A fine-tuned 7B model scoring inbound signals must return in under 200ms to feel synchronous in an ABM orchestration workflow. KV cache and Flash Attention are what make that latency achievable.

Let's build a production-grade cost estimator that accounts for both attention and MLP FLOPs, models KV cache memory pressure, and flags when a batch configuration will OOM.

```python
import torch
import time

class InferenceCostModel:
    def __init__(self, n_layers, n_heads, d_head, d_model, d_ff, vocab_size, dtype_bytes=2):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.dtype_bytes = dtype_bytes

    def kv_cache_bytes(self, seq_len, batch_size):
        per_token = 2 * self.n_layers * self.n_heads * self.d_head * self.dtype_bytes
        return per_token * seq_len * batch_size

    def attention_flops(self, prompt_len, gen_len):
        prefill = self.n_layers * self.n_heads * 2 * prompt_len * prompt_len * self.d_head
        decode = self.n_layers * self.n_heads * 2 * self.d_head * sum(
            prompt_len + t for t in range(gen_len)
        )
        return prefill + decode

    def mlp_flops(self, total_tokens):
        return self.n_layers * 2 * self.d_model * self.d_ff * total_tokens

    def total_flops(self, prompt_len, gen_len):
        total_tokens = (prompt_len + gen_len) * (gen_len + 1)
        return self.attention_flops(prompt_len, gen_len) + self.mlp_flops(total_tokens)

    def estimate_latency_ms(self, prompt_len, gen_len, batch_size, gpu_tflops=312, gpu_hbm_gb=80):
        total_flops = self.total_flops(prompt_len, gen_len) * batch_size
        compute_ms = total_flops / (gpu_tflops * 1e9)

        cache_bytes = self.kv_cache_bytes(prompt_len + gen_len, batch_size)
        cache_gb = cache_bytes / 1e9

        oom_risk = cache_gb > gpu_hbm_gb * 0.7

        return {
            "total_tflops": total_flops / 1e12,
            "compute_ms": compute_ms,
            "kv_cache_gb": cache_gb,
            "oom_risk": oom_risk,
            "max_safe_batch": int(gpu_hbm_gb * 0.7 / (cache_gb / batch_size)) if cache_gb > 0 else batch_size,
        }


llama_7b = InferenceCostModel(
    n_layers=32, n_heads=32, d_head=128, d_model=4096, d_ff=11008, vocab_size=32000
)

llama_70b = InferenceCostModel(
    n_layers=80, n_heads=64, d_head=128, d_model=8192, d_ff=28672, vocab_size=32000
)

print("=" * 80)
print("PRODUCTION INFERENCE COST ESTIMATOR")
print("=" * 80)

for name, model in [("Llama-2 7B", llama_7b), ("Llama-2 70B", llama_70b)]:
    print(f"\n{'=' * 80}")
    print(f"Model: {name}")
    print(f"{'=' * 80}")

    for desc, p, g, b in [
        ("ABM signal scoring\n  (128 tok, 16 tok, batch=32)", 128, 16, 32),
        ("SDR email generation\n  (800 tok, 200 tok, batch=8)", 800, 200, 8),
        ("Account research\n  (2000 tok, 500 tok, batch=4)", 2000, 500, 4),
    ]:
        r = model.estimate_latency_ms(p, g, b)
        flag = " *** OOM RISK ***" if r["oom_risk"] else ""
        print(f"\n  {desc}")
        print(f"    Total compute:     {r['total_tflops']:>10.1f} TFLOPs")
        print(f"    Est. GPU time:     {r['compute_ms']:>10.1f} ms")
        print(f"    KV cache memory:   {r['kv_cache_gb']:>10.2f} GB{flag}")
        print(f"    Max safe batch:    {r['max_safe_batch']:>10d}")

print("\n" + "=" * 80)
print("DEPLOYMENT DECISION RULES")
print("=" * 80)
print("""
1. If kv_cache_gb > 0.7 * GPU_HBM: reduce batch size or switch to paged
   attention (vLLM, SGLang).
2. If compute_ms > target_latency_ms: quantize to FP8 or INT8, or distill
   to a smaller model for the signal-scoring path.
3. For batch enrichment (not real-time): maximize batch size until KV cache
   hits 0.7 * HBM. Prefill is parallelizable; decode is the serial bottleneck.
4. For real-time scoring (ABM signal orchestration): keep prompt < 256 tokens
   and generation < 32 tokens. KV cache memory stays under 1 GB at batch 32.
""")
```

The decision rules at the bottom are the operational output of everything we built. Rule 4 is the one that connects most directly to Zone 7: a fine-tuned scoring model trained on deal history — where job changes, social signals, and events are labels — must serve predictions at ABM orchestration latency. That means short prompts, short generations, high batch. The inference math we just computed is the constraint that shapes the model architecture choice, the prompt template design, and the batch scheduling strategy for the entire signal pipeline.

---
## Ship It

Production deployment starts with the serving framework. vLLM implements speculative decoding via the `--speculative-model` flag, and as of vLLM 0.6+ supports EAGLE-style draft heads. The configuration specifies the draft model checkpoint, the number of speculative tokens (K), and optionally the tree topology. The tree-attention kernel must be compatible with your attention backend — FlashAttention-2 supports the tree-attention masks EAGLE-3 requires, but older backends may fall back to a less efficient implementation that negates the speedup.

```python
import subprocess
import json
import time
import requests

LAUNCH_CMD = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "meta-llama/Llama-3.1-8B-Instruct",
    "--speculative-model", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
    "--num-speculative-tokens", "4",
    "--use-v2-block-manager",
    "--port", "8001",
]

def benchmark_endpoint(url, num_requests=20, prompt="Explain what B2B lead enrichment means in 3 sentences."):
    latencies = []
    token_counts = []
    for i in range(num_requests):
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
        }
        t0 = time.perf_counter()
        resp = requests.post(f"{url}/v1/chat/completions", json=payload)
        t1 = time.perf_counter()
        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            latencies.append(t1 - t0)
            token_counts.append(completion_tokens)
        else:
            print(f"Request {i} failed: {resp.status_code} {resp.text[:200]}")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        avg_tokens = sum(token_counts) / len(token_counts)
        avg_tps = avg_tokens / avg_latency
        print(f"Requests:         {len(latencies)}")
        print(f"Avg latency:      {avg_latency:.3f}s")
        print(f"Avg tokens/resp:  {avg_tokens:.1f}")
        print(f"Avg tokens/sec:   {avg_tps:.1f}")
        return avg_tps
    return 0.0

print("To launch the speculative endpoint:")
print(" ".join(LAUNCH_CMD))
print()

MEASUREMENT_PLAN = {
    "step_1": "Sample 200 prompts from your production enrichment pipeline",
    "step_2": "Run through speculative endpoint at target concurrency (batch=32)",
    "step_3": "Run same prompts through non-speculative endpoint",
    "step_4": "Compare p50/p90/p99 latency and tokens-per-second",
    "step_5": "Extract alpha from vLLM metrics: spec_token_accept_rate",
    "gate": "Enable speculation only if mean_alpha > 0.55 AND p90_latency_improvement > 20%",
    "fallback": "If alpha drops below 0.5 in production, disable spec via config reload",
}

print("MEASUREMENT PLAN:")
for step, desc in MEASUREMENT_PLAN.items():
    print(f"  {step}: {desc}")

DASHBOARD_METRICS = {
    "vllm:spec_token_accept_rate": {
        "description": "Fraction of draft tokens accepted by target model",
        "alert_threshold": "< 0.6 for 5 consecutive minutes",
        "action": "Page on-call; consider disabling speculative decoding",
    },
    "vllm:spec_token_draft_accept_length": {
        "description": "Average number of accepted tokens per draft tree",
        "alert_threshold": "< 1.5 (near break-even)",
        "action": "Investigate prompt distribution shift",
    },
    "vllm:time_to_first_token_seconds": {
        "description": "TTFT should not increase with speculation enabled",
        "alert_threshold": "> baseline + 50ms",
        "action": "Check tree-attention kernel; draft model may be adding prefill latency",
    },
    "vllm:request_inference_duration_seconds": {
        "description": "End-to-end generation time per request",
        "alert_threshold": "> baseline (speculation should reduce this)",
        "action": "Disable speculation; alpha is too low for current traffic",
    },
}

print("\nDASHBOARD METRICS:")
for metric, config in DASHBOARD_METRICS.items():
    print(f"\n  {metric}")
    for k, v in config.items():
        print(f"    {k}: {v}")
```

```
To launch the speculative endpoint:
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-8B-Instruct --speculative-model yuhuili/EAGLE3-LLaMA3.1-Instruct-8B --num-speculative-tokens 4 --use-v2-block-manager --port 8001

MEASUREMENT PLAN:
  step_1: Sample 200 prompts from your production enrichment pipeline
  step_2: Run through speculative endpoint at target concurrency (batch=32)
  step_3: Run same prompts through non-speculative endpoint
  step_4: Compare p50/p90/p99 latency and tokens-per-second
  step_5: Extract alpha from vLLM metrics: spec_token_accept_rate
  gate: Enable speculation only if mean_alpha > 0.55 AND p90_latency_improvement > 20%
  fallback: If alpha drops below 0.5 in production, disable spec via config reload

DASHBOARD METRICS:

  vllm:spec_token_accept_rate
    description: Fraction of draft tokens accepted by target model
    alert_threshold: < 0.6 for 5 consecutive minutes
    action: Page on-call; consider disabling speculative decoding

  vllm:spec_token_draft_accept_length
    description: Average number of accepted tokens per draft tree
    alert_threshold: < 1.5 (near break-even)
    action: Investigigate prompt distribution shift

  vllm:time_to_first_token_seconds
    description: TTFT should not increase with speculation enabled
    alert_threshold: > baseline + 50ms
    action: Check tree-attention kernel; draft model may be adding prefill latency

  vllm:request_inference_duration_seconds
    description: End-to-end generation time per request
    alert_threshold: > baseline (speculation should reduce this)
    action: Disable speculation; alpha is too low for current traffic
```

Cost modeling is the final piece. Speculation trades FLOPs for latency: the draft head adds forward passes, and rejected drafts waste compute. At alpha=0.7 with K=4, you do approximately 1.5x more FLOPs per generated token (the target forward pass processes 4 candidate positions but only accepts ~2.5 on average). On a per-GPU basis, if your GPU is compute-bound (unlikely during decode) this is a net cost increase. If your GPU is memory-bandwidth-bound (the normal case during decode), the extra FLOPs are free because the compute units were idle anyway — you're trading unused compute capacity for real latency reduction. The break-even point depends on your concurrency: at low concurrency (batch size 1-4), speculation is almost always a win because the GPU is deeply underutilized. At high concurrency (batch size 64+), the GPU approaches compute saturation, and the extra draft FLOPs start competing with useful work, raising the break-even alpha.

In the GTM lifecycle framing — "MLOps for GTM = versioning your enrichment waterfalls, detecting when your scoring model drifts" — speculative decoding is an inference-config parameter that needs the same versioning and drift monitoring as your model weights. When your enrichment prompt templates change (new fields added, different summarization instructions), alpha can shift because the target model's hidden-state distribution shifts. The acceptance rate metric is your drift detector: a sudden drop after a prompt-template deploy means the draft head's training distribution no longer matches production traffic, and you need to either retrain the draft head or fall back to non-speculative serving until it catches up.
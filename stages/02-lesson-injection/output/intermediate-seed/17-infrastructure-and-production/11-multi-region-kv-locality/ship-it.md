## Ship It

Deploying a cache-aware multi-region inference setup requires three components: the inference engine with prefix caching enabled, a router that routes on prefix-hash match, and a DR plan that does not lose your tokenizer.

**Component 1: Enable prefix caching in vLLM.** Start each regional vLLM instance with `--enable-prefix-caching`. This enables block-level KV cache reuse. There is no configuration for cache size — vLLM uses available HBM after model weights are loaded. Monitor `vllm:gpu_cache_usage_perc` to ensure you are not evicting useful prefixes due to memory pressure.

**Component 2: Deploy a cache-aware router.** vLLM Router (Rust) consumes KV-cache events from each backend node and routes on prefix-hash match. Configure it with your backend node list and set the tie-breaker to GPU utilization. The router's routing decision is the longest common prefix between the incoming request and any cached prefix on any node. For cross-region deployments, add a latency penalty term: if the best cache match is in a region with >150 ms network latency, compare the cache savings against the network cost and fall back to local prefill if network cost exceeds cache savings.

```python
import time
import hashlib

class ProductionRouterDecision:
    def __init__(self, nodes, prefill_ms_per_token=0.4, network_penalty_per_ms=1.0):
        self.nodes = nodes
        self.prefill_ms_per_token = prefill_ms_per_token
        self.network_penalty_per_ms = network_penalty_per_ms

    def evaluate_routing_decision(self, tokens, source_region):
        decisions = []
        total_tokens = len(tokens)

        for node in self.nodes:
            match = node.longest_prefix_match(tokens)
            cached_tokens = match * BLOCK_SIZE
            prefill_tokens = total_tokens - cached_tokens

            prefill_cost = prefill_tokens * self.prefill_ms_per_token

            latency_map = {
                ("us-east-1", "eu-west-1"): 70,
                ("us-east-1", "ap-southeast-1"): 180,
                ("eu-west-1", "ap-southeast-1"): 250,
            }
            latency_map[("us-east-1", "us-east-1")] = 1
            latency_map[("eu-west-1", "eu-west-1")] = 1
            latency_map[("ap-southeast-1", "ap-southeast-1")] = 1

            for (r1, r2), lat in latency_map.items():
                if source_region in (r1, r2) and node.region in (r1, r2):
                    network_latency = lat
                    break
            else:
                network_latency = 200

            total_cost = prefill_cost + network_latency

            decisions.append({
                "node": node.name,
                "region": node.region,
                "cached_tokens": cached_tokens,
                "prefill_tokens": prefill_tokens,
                "prefill_cost_ms": prefill_cost,
                "network_latency_ms": network_latency,
                "total_cost_ms": total_cost,
                "is_cache_hit": cached_tokens > 0,
            })

        decisions.sort(key=lambda d: d["total_cost_ms"])
        best = decisions[0]

        print(f"\n  Source region: {source_region}")
        print(f"  Total tokens:  {total_tokens}")
        print(f"  {'Node':<18} {'Region':<16} {'Cached':<8} {'Prefill':<8} {'Net(ms)':<9} {'Total(ms)':<10} {'Decision'}")
        print(f"  {'-'*85}")

        for d in decisions:
            marker = " <== SELECTED" if d == best else ""
            hit = "HIT" if d["is_cache_hit"] else "MISS"
            print(f"  {d['node']:<18} {d['region']:<16} {d['cached_tokens']:<8} {d['prefill_tokens']:<8} {d['network_latency_ms']:<9} {d['total_cost_ms']:<10.1f} {hit}{marker}")

        cache_savings = (total_tokens - best["prefill_tokens"]) * self.prefill_ms_per_token
        print(f"\n  Selected: {best['node']} ({best['region']})")
        print(f"  Cache savings: {cache_savings:.0f} ms avoided prefill")
        print(f"  Net total cost: {best['total_cost_ms']:.0f} ms (prefill {best['prefill_cost_ms']:.0f} + network {best['network_latency_ms']} ms)")
        return best

nodes = [
    InferenceNode("infer-us-1", "us-east-1"),
    InferenceNode("infer-eu-1", "eu-west-1"),
    InferenceNode("infer-ap-1", "ap-southeast-1"),
]

shared_system_prompt = [hash(f"sys_{i}") % 32000 for i in range(512)]
for node in nodes[:2]:
    node.cache_request(shared_system_prompt)

router = ProductionRouterDecision(nodes)

print("=" * 90)
print("ROUTING DECISION: Request from Frankfurt (eu-west-1), shared system prompt")
print("=" * 90)
router.evaluate_routing_decision(
    shared_system_prompt + [hash(f"user_{i}") % 32000 for i in range(128)],
    "eu-west-1"
)

print("\n" + "=" * 90)
print("ROUTING DECISION: Request from Singapore (ap-southeast-1), shared system prompt")
print("=" * 90)
router.evaluate_routing_decision(
    shared_system_prompt + [hash(f"user_{i}") % 32000 for i in range(128)],
    "ap-southeast-1"
)

print("\n" + "=" * 90)
print("ROUTING DECISION: Request from Frankfurt, NO shared prefix (cold)")
print("=" * 90)
router.evaluate_routing_decision(
    [hash(f"new_{i}") % 32000 for i in range(640)],
    "eu-west-1"
)
```

**Component 3: DR checklist.** 32% of LLM DR failures occur because teams backed up model weights but forgot the tokenizer files or quantization configs. JPMorgan and Mayo Clinic ran a cross-region failover test in November 2024 that took ~22 minutes. The failures were not GPU provisioning — they were missing artifacts. Your DR checklist must include: (1) `model.bin` / `model.safetensors` (weights), (2) `tokenizer.json` + `tokenizer_config.json` + `special_tokens_map.json` (tokenizer), (3) `config.json` + generation config (model configuration), (4) `quantization_config.json` if using AWQ/GPTQ. Without all four, the model will not load.

In GTM terms, this maps to Zone 17 — the MLOps/GTM system lifecycle. Versioning your enrichment waterfalls and detecting scoring drift is the GTM equivalent of versioning your inference artifacts. If you deploy a new Clay table version that calls a different LLM endpoint, you need the same artifact discipline: the table definition, the enrichment formulas, the scoring model config, and the prompt template. A Clay table without its enrichment formulas is a model without its tokenizer — it loads but produces wrong output. [CITATION NEEDED — concept: Clay table artifact dependencies for DR]

```python
import os
import json

def generate_dr_checklist(model_name, artifact_paths):
    required = {
        "weights": ["model.safetensors", "model.bin", "pytorch_model.bin"],
        "tokenizer": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "config": ["config.json", "generation_config.json"],
        "quantization": ["quantize_config.json", "quantization_config.json"],
    }

    print(f"\n{'='*65}")
    print(f"  DR Checklist: {model_name}")
    print(f"{'='*65}")

    all_present = True
    for category, filenames in required.items():
        found = []
        missing = []
        for fn in filenames:
            present = any(fn in str(p) for p in artifact_paths)
            if present:
                found.append(fn)
            else:
                missing.append(fn)

        if found:
            status = "OK"
            detail = f"Found: {found[0]}"
        elif missing:
            if category == "quantization":
                status = "SKIP"
                detail = "Not quantized (or check manually)"
            else:
                status = "MISSING"
                detail = f"Missing: {', '.join(missing)}"
                all_present = False

        marker = "[OK]" if status == "OK" else ("[--]" if status == "SKIP" else "[!!]")
        print(f"  {marker} {category:<15} {detail}")

    can_restore = all_present
    print(f"\n  Restore ready: {'YES' if can_restore else 'NO — fix missing artifacts'}")
    if not can_restore:
        print(f"  Risk: 32% of LLM DR failures are from missing non-weight artifacts")
    print(f"{'='*65}")
    return can_restore

artifacts_good = [
    "/models/llama-3-70b/model.safetensors",
    "/models/llama-3-70b/tokenizer.json",
    "/models/llama-3-70b/tokenizer_config.json",
    "/models/llama-3-70b/special_tokens_map.json",
    "/models/llama-3-70b/config.json",
    "/models/llama-3-70b/generation_config.json",
]

artifacts_bad = [
    "/models/llama-3-70b/model.safetensors",
    "/models/llama-3-70b/config.json",
]

generate_dr_checklist("Llama-3-70B (complete)", artifacts_good)
generate_dr_checklist("Llama-3-70B (missing tokenizer)", artifacts_bad)
```
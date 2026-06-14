# Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang

## Learning Objectives

- Compare five inference engines by the mechanism each optimizes for (batching strategy, KV cache management, quantization approach, prefix deduplication).
- Select an engine given hardware constraints (CPU-only, AMD, NVIDIA Hopper/Blackwell), concurrency requirements (1 user vs. 10,000), and workload shape (general chat, agentic multi-turn, shared-prefix batch jobs).
- Launch a vLLM server with memory-constrained flags and measure throughput scaling across concurrency levels.
- Configure SGLang prefix caching and compute KV cache reuse percentage from server logs.
- Explain why TGI's December 2025 maintenance-mode status biases new projects toward vLLM or SGLang.

## The Problem

Your team starts a new self-hosted LLM project. One engineer says Ollama because it installed in two minutes on their laptop. Another says vLLM because they read about PagedAttention. A third says TGI because the Hugging Face ecosystem feels safe. Meanwhile, your GTM pipeline needs to enrich 40,000 accounts with proprietary firmographic data that cannot leave your VPC, and the choice you make right now determines whether that pipeline finishes in four hours or four days.

The real problem is that these five tools are not competing implementations of the same idea. They optimize for different bottlenecks. llama.cpp optimizes for the case where you have no GPU. Ollama optimizes for the case where you have no patience for compilation. TGI optimized for the case where you wanted Hugging Face integration with solid observability. vLLM optimizes for GPU memory fragmentation under concurrent load. SGLang optimizes for redundant computation across requests that share prompt prefixes. Picking the wrong bottleneck to solve means you either waste GPU hours on a server that can't batch, or you over-engineer a GPU cluster for a workload that would have run fine on a laptop.

This is also an MLOps-for-GTM decision, not just an infrastructure decision. Zone 17 of the GTM lifecycle — versioning your enrichment waterfalls, detecting when your scoring model drifts — depends on reproducible inference. If your dev environment runs Ollama with a 4-bit GGUF quantization of Llama 3.1 8B and your production environment runs vLLM with the FP16 HF weights, you have a silent scoring drift problem. Your enrichment outputs in staging will not match production, and you will not know why. The same weights across the pipeline is not a nice-to-have; it is a correctness requirement for any GTM system that feeds scoring models.

## The Concept

Five mechanisms separate these engines. Every feature list you will read online is a manifestation of one of these five underlying choices.

**Batching strategy.** Static batching collects N requests, waits until the batch is full or a timeout fires, processes all N sequences to completion, then starts the next batch. If one sequence finishes early, its GPU slot sits idle until the longest sequence in the batch finishes. Continuous batching (sometimes called in-flight batching or iteration-level scheduling) solves this: at every decoding step, the scheduler checks whether any sequence has finished. If one has, it immediately inserts a new request from the queue into that freed slot. This is why vLLM and SGLang serve an order of magnitude more concurrent requests than naive implementations — they never leave GPU cycles idle waiting for a batch to drain. llama.cpp uses static batching. TGI, vLLM, and SGLang all implement continuous batching, but with different schedulers and memory backends.

**KV cache memory management.** During autoregressive generation, each token's attention computation requires access to the key-value tensors of every previous token in the sequence. These KV caches grow linearly with sequence length and consume significant VRAM. The naive approach pre-allocates a contiguous block of memory sized for the maximum possible sequence length — which means if your max length is 4096 tokens but the average request uses 800, you waste 80% of your KV cache memory to fragmentation and over-allocation. vLLM's PagedAttention borrows from operating system virtual memory: it partitions the KV cache into fixed-size blocks (like memory pages) and maintains a block table mapping logical token positions to physical blocks. Sequences that share a prefix point to the same physical blocks. Memory is allocated on demand as sequences grow. This eliminates fragmentation and enables the prefix-sharing optimization. SGLang's RadixAttention takes this further by maintaining a radix tree of KV cache blocks indexed by token prefix, so when 50 GTM enrichment requests all start with the same 500-token system prompt, that prefix is computed once and reused across all 50.

**Quantization at the metal.** llama.cpp runs GGUF-format models where weights are pre-quantized to 4-bit, 5-bit, or 8-bit integers. During the matrix multiply, weights are dequantized on-the-fly into the native precision of the compute unit. This is weight-only quantization — activations remain in higher precision. The trade-off is speed (dequantization adds overhead per operation) versus memory (4-bit weights occupy one-eighth the memory of FP32). On CPU, this is the only viable path for running a 7B model in 4GB of RAM. On GPU, GGUF quantization trades inference quality for the ability to fit larger models on smaller cards. Ollama uses this same mechanism under the hood — it ships llama.cpp compiled as a shared library, wrapped in a Go HTTP server.

**Scheduling primitives for shared prefixes.** If your GTM workflow sends 1,000 personalized outreach emails where each request is `[system prompt: 800 tokens] + [account context: 300 tokens] + [instruction: 50 tokens]`, the first 800 tokens are identical across all 1,000 requests. A naive server computes those 800 tokens of KV cache 1,000 times. SGLang's RadixAttention computes them once and stores the KV cache in a radix tree keyed by token sequence. Each subsequent request traverses the tree, finds the matching prefix, and starts computation from token 801. For workloads with high prefix overlap, this is not a 10-20% optimization — it can be a 5-10x reduction in prefill compute.

**Abstraction layer.** Ollama is not an inference engine. It is a distribution and management layer around llama.cpp. It solves the problem of "I want to run a local model without compiling C++, managing GGUF files, or writing my own HTTP wrapper." It exposes an OpenAI-compatible API, handles model pulling from a registry, and manages model lifecycle (load/unload from memory). The cost of this abstraction is roughly 15-30% throughput overhead compared to raw llama.cpp, due to Go-to-CGo boundary crossing and HTTP serialization. For development and prototyping, this is irrelevant. For production at scale, it compounds.

```mermaid
flowchart TD
    A["New self-hosted inference project"] --> B{"Hardware?'}
    B -->|'CPU-only or consumer GPU'| C['llama.cpp / Ollama']
    B -->|'AMD GPU'| D['vLLM only\nTRT-LLM is NVIDIA-locked']
    B -->|'NVIDIA datacenter GPU'| E{'Concurrency need?'}
    E -->|'Low (<10 reqs)'| F['llama.cpp server\nor Ollama']
    E -->|'Moderate (10-100)'| G{'TGI status?'}
    G -->|'Dec 2025: maintenance mode'| H['Avoid for new projects']
    E -->|'High (100-10,000+)'| I{'Workload shape?'}
    I -->|'General chat / completion'| J['vLLM\nPagedAttention + continuous batching']
    I -->|'Shared prompt prefixes\nagentic multi-turn'| K['SGLang\nRadixAttention prefix deduplication']
    C --> L['Pipeline: dev=Ollama\nstaging=llama.cpp\nprod=vLLM or SGLang']
    D --> L
    F --> L
    J --> L
    K --> L
```

Here is a decision-tree walker that encodes this logic. Run it and it prints a recommendation given your constraints:

```python
import json

def select_engine(hardware, concurrency, workload, new_project=True):
    if hardware == "cpu":
        return {
            "engine": "llama.cpp",
            "reason": "CPU-only computation. Weight-only GGUF quantization with on-the-fly dequantization. No other engine runs efficiently without a GPU.",
            "format": "GGUF (4-bit, 5-bit, or 8-bit)",
        }
    
    if hardware == "amd":
        return {
            "engine": "vLLM",
            "reason": "AMD ROCm support. TRT-LLM is NVIDIA-locked. llama.cpp has limited AMD GPU support. vLLM is the only production-grade option for AMD.",
            "format": "HF safetensors (FP16 or BF16)",
        }
    
    if concurrency < 10:
        return {
            "engine": "llama.cpp server or Ollama",
            "reason": "Low concurrency does not justify the memory overhead of continuous batching infrastructure. Static batching is sufficient.",
            "format": "GGUF for llama.cpp, auto-pulled by Ollama",
        }
    
    if not new_project and hardware == "nvidia" and concurrency < 500:
        return {
            "engine": "TGI (legacy only)",
            "reason": "TGI entered maintenance mode December 11, 2025. Acceptable for existing deployments. Not recommended for new projects.",
            "format": "HF safetensors",
        }
    
    if "shared_prefix" in workload or "agentic" in workload:
        return {
            "engine": "SGLang",
            "reason": "RadixAttention deduplicates KV cache across requests sharing prompt prefixes. 5-10x prefill reduction for high-overlap workloads.",
            "format": "HF safetensors",
        }
    
    return {
        "engine": "vLLM",
        "reason": "PagedAttention eliminates KV cache fragmentation. Continuous batching maximizes GPU utilization under concurrent load. Production default for general-purpose inference.",
        "format": "HF safetensors (FP16, BF16, or AWQ/GPTQ quantized)",
    }


scenarios = [
    ("MacBook Pro, no GPU", "cpu", 1, "local_chat", True),
    ("AMD MI300X cluster", "amd", 1000, "general", True),
    ("Dev laptop, A100", "nvidia", 3, "prototyping", True),
    ("Existing TGI deployment", "nvidia", 200, "general", False),
    ("GTM enrichment, 32 shared-prefix reqs", "nvidia", 500, "shared_prefix", True),
    ("General production chatbot", "nvidia", 2000, "general", True),
]

for label, hw, conc, wl, new in scenarios:
    result = select_engine(hw, conc, wl, new)
    print(f"Scenario: {label}")
    print(f"  Recommended: {result['engine']}")
    print(f"  Why: {result['reason']}")
    print(f"  Format: {result['format']}")
    print()
```

When you run this, you get structured recommendations. The decision tree is intentionally simple because the mechanisms are simple — the complexity people attribute to "choosing an inference engine" usually comes from reading feature lists without understanding which mechanism each feature implements.

One status note that affects every decision in this space: TGI entered maintenance mode on December 11, 2025. Hugging Face announced that only bug fixes will follow — no new features, no architecture-level optimizations. This does not mean existing TGI deployments are broken. It means if you start a new project today on TGI, you are betting on a platform that will not receive performance improvements or support for new model architectures beyond what already exists. vLLM and SGLang are actively developed and are the safer default for new work.

## Build It

The benchmark script below launches requests against any running OpenAI-compatible server (Ollama, vLLM, SGLang, llama.cpp server, or TGI) and measures three things: time-to-first-token, tokens per second at steady state, and how throughput degrades as concurrency increases. These three numbers make the mechanism differences visible. A static-batching server will show flat or degrading throughput as concurrency rises. A continuous-batching server with PagedAttention will show near-linear scaling until GPU compute saturates. A prefix-caching server will show dramatically lower TTFT on requests that share prefixes.

```python
import asyncio
import time
import json
import aiohttp

async def stream_completion(session, base_url, prompt, model, max_tokens=128):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    
    ttft = None
    token_count = 0
    start = time.monotonic()
    
    try:
        async with session.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices") and chunk["choices"][0].get("text"):
                        if ttft is None:
                            ttft = time.monotonic() - start
                        token_count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e), "ttft": None, "tokens": 0, "elapsed": 0}
    
    elapsed = time.monotonic() - start
    tps = token_count / elapsed if elapsed > 0 else 0
    
    return {
        "ttft": round(ttft, 4) if ttft else None,
        "tokens": token_count,
        "elapsed": round(elapsed, 4),
        "tps": round(tps, 2),
    }

async def benchmark(base_url, model, concurrency_levels, prompt, max_tokens=64):
    results = {}
    async with aiohttp.ClientSession() as session:
        for c in concurrency_levels:
            tasks = [
                stream_completion(session, base_url, prompt, model, max_tokens)
                for _ in range(c)
            ]
            start = time.monotonic()
            responses = await asyncio.gather(*tasks)
            wall_time = time.monotonic() - start
            
            errors = [r for r in responses if "error" in r]
            valid = [r for r in responses if "error" not in r]
            
            if valid:
                avg_ttft = sum(r["ttft"] or 0 for r in valid) / len(valid)
                avg_tps = sum(r["tps"] for r in valid) / len(valid)
                total_tokens = sum(r["tokens"] for r in valid)
                aggregate_tps = total_tokens / wall_time if wall_time > 0 else 0
            else:
                avg_ttft = 0
                avg_tps = 0
                total_tokens = 0
                aggregate_tps = 0
            
            results[c] = {
                "avg_ttft_s": round(avg_ttft, 4),
                "avg_per_req_tps": round(avg_tps, 2),
                "aggregate_tps": round(aggregate_tps, 2),
                "total_tokens": total_tokens,
                "wall_time_s": round(wall_time, 2),
                "errors": len(errors),
            }
            
            print(f"  Concurrency {c:>3}: TTFT={avg_ttft:.4f}s  "
                  f"per-req={avg_tps:.1f} tok/s  "
                  f"aggregate={aggregate_tps:.1f} tok/s  "
                  f"wall={wall_time:.2f}s  errors={len(errors)}")
    
    return results

async def main():
    import sys
    
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"
    
    prompt = "List five characteristics of high-performing go-to-market teams. Be specific and concise."
    concurrency_levels = [1, 4, 8]
    
    print(f"Server: {base_url}")
    print(f"Model:  {model}")
    print(f"Prompt: {prompt[:60]}...")
    print(f"Max tokens per request: 64")
    print()
    print("Running benchmarks...")
    
    results = await benchmark(base_url, model, concurrency_levels, prompt)
    
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Concurrency':>12} | {'Avg TTFT':>10} | {'Per-Req TPS':>12} | {'Aggregate TPS':>14} | {'Errors':>7}")
    print("-" * 72)
    for c, r in results.items():
        print(f"{c:>12} | {r['avg_ttft_s']:>9.4f}s | {r['avg_per_req_tps']:>12.1f} | {r['aggregate_tps']:>14.1f} | {r['errors']:>7}")
    print()
    
    c1 = results.get(1, {})
    c8 = results.get(8, {})
    if c1.get("aggregate_tps", 0) > 0 and c8.get("aggregate_tps", 0) > 0:
        ratio = c8["aggregate_tps"] / c1["aggregate_tps"]
        print(f"Scaling ratio (conc-8 / conc-1): {ratio:.2f}x")
        if ratio < 1.5:
            print("  -> Poor scaling. Likely static batching or compute-bound.")
        elif ratio < 4.0:
            print("  -> Moderate scaling. Continuous batching helping but hitting limits.")
        else:
            print("  -> Strong scaling. Continuous batching + efficient memory management.")

if __name__ == "__main__":
    asyncio.run(main())
```

Save this as `bench.py`. Point it at any running server:

```
python bench.py http://localhost:11434 llama3.1:8b
```

The output is a table of numbers. No claims, no assertions — just TTFT, per-request throughput, aggregate throughput, and error counts at each concurrency level. The scaling ratio at the end tells you what mechanism is dominant: if aggregate throughput at concurrency 8 is less than 1.5x the throughput at concurrency 1, the server is either using static batching (each request competes for the same GPU) or is compute-bound (the GPU is already saturated by a single request). If the ratio is 4x or higher, continuous batching and efficient memory management are doing their job.

## Use It

This is where Zone 17 of the GTM lifecycle becomes real. Zone 17 maps MLOps and model lifecycle to GTM system lifecycle: versioning your enrichment waterfalls, detecting when your scoring model drifts, retraining when the ground truth shifts. Self-hosted serving is the infrastructure layer that makes all of that possible when your data cannot leave your VPC.

Consider a concrete GTM workflow: account scoring across 40,000 companies using proprietary firmographic data. Each account gets a prompt like `[system: scoring rubric] + [account data: revenue, headcount, tech stack] + [instruction: score 1-10 and explain]`. If this data includes Salesforce records, product usage telemetry, or contract terms, it cannot go to OpenAI or Anthropic. You need to serve the model yourself.

The engine choice here is driven by workload shape. Those 40,000 requests share a system prompt (the scoring rubric) that is 500-1000 tokens long. A naive server recomputes the KV cache for that prefix 40,000 times. SGLang's RadixAttention computes it once. The difference is not theoretical — measure it:

```python
import asyncio
import aiohttp
import time
import json

SHARED_PREFIX = """You are an expert account scoring analyst for a B2B SaaS company.
Your task is to evaluate each account on a scale of 1-10 based on:
1. Firmographic fit (revenue band, employee count, industry)
2. Technology stack alignment with our product
3. Engagement signals (website visits, content downloads, event attendance)
4. Competitive presence (do they use a competitor's product?)

Respond with a JSON object containing 'score' (1-10) and 'reasoning' (one sentence).
Always respond with valid JSON only.""" + "\n\n"

ACCOUNTS = [
    "Account: TechFlow Inc. Revenue: $50M. Employees: 250. Stack: AWS, Snowflake, dbt. Engagement: 3 site visits. Competitor: none.",
    "Account: DataPipe LLC. Revenue: $20M. Employees: 120. Stack: GCP, BigQuery. Engagement: 1 download. Competitor: Fivetran.",
    "Account: CloudVista Corp. Revenue: $100M. Employees: 500. Stack: Azure, Databricks. Engagement: attended webinar. Competitor: none.",
    "Account: ByteForge Inc. Revenue: $5M. Employees: 30. Stack: AWS, Redshift. Engagement: none. Competitor: Hevo.",
    "Account: ScaleMatrix LLC. Revenue: $80M. Employees: 400. Stack: multi-cloud, Snowflake, Airflow. Engagement: 5 site visits, 2 downloads. Competitor: none.",
] * 6

async def score_account(session, base_url, model, account_text):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SHARED_PREFIX},
            {"role": "user", "content": account_text},
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }
    start = time.monotonic()
    try:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            data = await resp.json()
            elapsed = time.monotonic() - start
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"elapsed": round(elapsed, 4), "content": content[:80]}
    except Exception as e:
        return {"elapsed": round(time.monotonic() - start, 4), "error": str(e)[:80]}

async def main():
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model = sys.argv[2] if len(sys.argv) > 2 else "llama3.1:8b"
    
    async with aiohttp.ClientSession() as session:
        print(f"Benchmarking {len(ACCOUNTS)} account-scoring requests")
        print(f"Shared system prompt: {len(SHARED_PREFIX)} chars")
        print(f"Server: {base_url}  Model: {model}")
        print()
        
        start = time.monotonic()
        results = await asyncio.gather(*[
            score_account(session, base_url, model, acct)
            for acct in ACCOUNTS
        ])
        wall = time.monotonic() - start
        
        valid = [r for r in results if "error" not in r]
        errors = [r for r in results if "error" in r]
        
        print(f"{'#':>3} | {'Latency':>8} | Response")
        print("-" * 72)
        for i, r in enumerate(valid[:10]):
            print(f"{i+1:>3} | {r['elapsed']:>7.3f}s | {r['content']}")
        if len(valid) > 10:
            print(f"... and {len(valid) - 10} more")
        print()
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(valid)}  Errors: {len(errors)}")
        print(f"Wall time: {wall:.2f}s")
        print(f"Effective throughput: {len(valid) / wall:.1f} accounts/sec")
        print(f"Estimated time for 40,000 accounts: {40000 / (len(valid) / wall) / 60:.1f} minutes")
        
        if errors:
            print(f"\nFirst error: {errors[0]['error']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run this against any server. The output tells you exactly how long 40,000 accounts would take at your current throughput. Against an Ollama dev server with static batching, you might see 2-3 accounts per second — which means 40,000 accounts takes 3-5 hours. Against vLLM with continuous batching on an A100, you might see 50-100 accounts per second — 7-13 minutes. Against SGLang with RadixAttention exploiting the shared 500-token scoring rubric prefix, you could see 150-200 accounts per second because the prefill cost of that shared prefix is amortized to near-zero across all requests. The mechanism difference translates directly into GTM operational reality: same enrichment job, 20x faster, same hardware.

This connects to the Zone 17 concept of versioning your enrichment waterfalls. When you version the waterfall, you version the model, the prompt, and the inference engine configuration. If your scoring drifts between versions, you need to know whether the cause is a prompt change, a model weight update, a quantization difference between GGUF and FP16, or a batching strategy change that affected latency-sensitive scoring behavior. Running the same weights across dev/staging/prod — GGUF in dev for prototyping, same GGUF or HF FP16 in staging, HF FP16 in production — narrows the debugging surface.

## Ship It

Production deployment of a self-hosted inference server follows a checklist that prevents the most common failures. None of these are theoretical — each one corresponds to an incident pattern observed in production GTM pipelines.

**GPU memory utilization.** vLLM's `--gpu-memory-utilization` flag controls what fraction of total VRAM the server will claim. Setting it to 0.9 means the server grabs 90% of VRAM for KV cache and model weights, leaving 10% for framework overhead. If you set it too high on a GPU that also runs CUDA context for other processes, you get OOM errors mid-request. Set it to 0.85 for a dedicated inference GPU, 0.7 if the GPU is shared. SGLang accepts the same flag with the same semantics.

**Max model length.** The `--max-model-len` flag sets the maximum sequence length the server will accept. This directly controls KV cache size — each token position requires approximately `2 * num_layers * num_kv_heads * head_dim * bytes_per_element` of KV cache memory. For Llama 3.1 8B, that is roughly 160KB per token at FP16. A max length of 8192 requires about 1.3GB of KV cache per concurrent sequence. Set this to the actual maximum your GTM prompts require, not the model's theoretical maximum. If your enrichment prompts are never longer than 2000 tokens, setting max-model-len to 8192 wastes VRAM that could serve more concurrent requests.

**Quantization consistency.** If you develop against Ollama running a 4-bit GGUF quantization of Llama 3.1 8B, your outputs reflect the quality degradation of 4-bit quantization. In production on vLLM running FP16 weights, those outputs will differ — sometimes subtly, sometimes materially for scoring tasks near decision boundaries. The fix is to use the same quantization level across environments, or to explicitly document the expected quality delta and validate it before shipping. For GTM scoring workflows where a 0.3-point score difference changes whether an account gets routed to sales, this matters.

Here is a pre-flight check script that validates your server configuration before you point a production pipeline at it:

```python
import asyncio
import aiohttp
import json
import time
import sys

async def preflight(base_url, model, expected_max_len=4096):
    checks = []
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{base_url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                models = await resp.json()
                model_ids = [m.get("id", "") for m in models.get("data", [])]
                if model in model_ids or any(model in mid for mid in model_ids):
                    checks.append(("Model loaded", "PASS", f"'{model}' found in server"))
                else:
                    checks.append(("Model loaded", "FAIL", f"'{model}' not found. Available: {model_ids[:5]}"))
        except Exception as e:
            checks.append(("Server reachable", "FAIL", str(e)[:80]))
            print_results(checks)
            return
        
        checks.append(("Server reachable", "PASS", f"{base_url} responded"))
        
        test_prompt = "Respond with exactly: PREFLIGHT_OK"
        try:
            async with session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": test_prompt,
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                content = data.get("choices", [{}])[0].get("text", "")
                if "PREFLIGHT" in content.upper():
                    checks.append(("Inference working", "PASS", f"Response: {content.strip()[:40]}"))
                else:
                    checks.append(("Inference working", "WARN", f"Unexpected response: {content.strip()[:40]}"))
                
                usage = data.get("usage", {})
                checks.append(("Token counting", "PASS" if usage else "WARN",
                              f"prompt={usage.get('prompt_tokens', '?')}, "
                              f"completion={usage.get('completion_tokens', '?')}"))
        except Exception as e:
            checks.append(("Inference working", "FAIL", str(e)[:80]))
        
        latency_times = []
        for i in range(3):
            start = time.monotonic()
            try:
                async with session.post(
                    f"{base_url}/v1/completions",
                    json={
                        "model": model,
                        "prompt": f"Count to {i+3}.",
                        "max_tokens": 20,
                        "temperature": 0.0,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    await resp.json()
                    latency_times.append(time.monotonic() - start)
            except Exception:
                pass
        
        if latency_times:
            avg_lat = sum(latency_times) / len(latency_times)
            checks.append(("Avg latency (3 requests)", "INFO", f"{avg_lat:.3f}s"))
            if avg_lat > 5.0:
                checks.append(("Latency assessment", "WARN", 
                              f"Average {avg_lat:.1f}s — check GPU utilization and batch settings"))
            else:
                checks.append(("Latency assessment", "PASS", f"Average {avg_lat:.3f}s is reasonable"))
        
        long_prompt = "word " * 100
        try:
            async with session.post(
                f"{base_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": long_prompt + "\nSummarize the above in one word.",
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                usage = data.get("usage", {})
                pt = usage.get("prompt_tokens", 0)
                if pt > 0:
                    checks.append(("Long context handling", "PASS", f"Accepted {pt} input tokens"))
                else:
                    checks.append(("Long context handling", "WARN", "No token count returned"))
        except Exception as e:
            checks.append(("Long context handling", "FAIL", str(e)[:80]))
    
    print_results(checks)

def print_results(checks):
    print("=" * 68)
    print("PREFLIGHT CHECK RESULTS")
    print("=" * 68)
    print(f"{'Check':<35} | {'Status':>6} | Detail")
    print("-" * 68)
    passes = 0
    warns = 0
    fails = 0
    for name, status, detail in checks:
        icon = {"PASS": "[OK]", "FAIL": "[XX]", "WARN": "[!!]", "INFO": "[--]"}[status]
        print(f"{name:<35} | {icon:>6} | {detail}")
        if status == "PASS":
            passes += 1
        elif status == "WARN":
            warns += 1
        elif status == "FAIL":
            fails += 1
    print("-" * 68)
    print(f"Summary: {passes} passed, {warns} warnings, {fails} failures")
    if fails > 0:
        print
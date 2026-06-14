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
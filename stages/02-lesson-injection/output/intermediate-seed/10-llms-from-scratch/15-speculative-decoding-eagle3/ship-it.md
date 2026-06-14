## Ship It

Production deployment of speculative decoding in 2025-2026 is dominated by vLLM and HuggingFace TGI, both of which implement EAGLE-style or Medusa-style speculative decoding as a serving-level feature. You do not write the rejection sampling loop yourself — the inference server handles draft proposal, tree verification, KV cache rollback, and bonus token emission internally. Your job is to configure it correctly and measure whether it actually helps.

The configuration surface in vLLM is straightforward. You pass a `--speculative-model` argument pointing to a draft model checkpoint, set `--num-speculative-tokens` to control draft length `N`, and vLLM handles the rest. The draft model must be compatible with the target model — same tokenizer, same vocabulary, and (for EAGLE-style drafters) trained on the target model's hidden states. The vLLM project maintains a list of compatible draft models for popular target models. [CITATION NEEDED — concept: vLLM speculative decoding configuration and compatible draft model list]

```python
import subprocess
import time
import requests
import json

CONFIG = {
    "target_model": "meta-llama/Llama-3.1-8B-Instruct",
    "draft_model": "meta-llama/Llama-3.2-1B-Instruct",
    "num_spec_tokens": 5,
    "port": 8000,
}

def check_vllm_available():
    try:
        result = subprocess.run(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"vLLM version: {result.stdout.strip()}")
            return True
        else:
            print("vLLM not installed. Install with: pip install vllm")
            return False
    except Exception:
        print("vLLM not installed. Install with: pip install vllm")
        return False

def build_launch_command(config, speculative=True):
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["target_model"],
        "--port", str(config["port"]),
    ]
    if speculative:
        cmd.extend([
            "--speculative-model", config["draft_model"],
            "--num-speculative-tokens", str(config["num_spec_tokens"]),
        ])
    return cmd

def print_launch_instructions(speculative=True):
    cmd = build_launch_command(CONFIG, speculative)
    label = "WITH speculative decoding" if speculative else "WITHOUT speculative decoding"
    print(f"\n{'=' * 60}")
    print(f"Launch command ({label}):")
    print(f"{'=' * 60}")
    print(" ".join(cmd))
    print()

def benchmark_prompt(prompt, max_tokens=100, port=8000):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": CONFIG["target_model"],
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            output = data["choices"][0]["text"]
            usage = data.get("usage", {})
            print(f"Prompt:     {prompt[:80]}...")
            print(f"Output:     {output[:80]}...")
            print(f"Tokens gen: {usage.get('completion_tokens', 'N/A')}")
            print(f"Wall time:  {elapsed:.3f}s")
            tok_rate = usage.get('completion_tokens', max_tokens) / elapsed
            print(f"Tokens/sec: {tok_rate:.1f}")
            return elapsed, tok_rate
        else:
            print(f"Error {resp.status_code}: {resp.text[:200]}")
            return None, None
    except requests.ConnectionError:
        print("Server not running. Start it with the launch command above.")
        return None, None

ENRICHMENT_PROMPT = (
    "Classify the following company into one of these categories: "
    "SaaS, Infrastructure, Fintech, Healthcare, E-commerce, Other. "
    "Respond with only the category name.\n\n"
    "Company: A cloud-native platform that provides real-time data "
    "pipelines and event streaming for enterprise data teams, "
    "with managed connectors and schema registry.\n"
    "Category:"
)

available = check_vllm_available()
print_launch_instructions(speculative=True)
print_launch_instructions(speculative=False)

print(f"\n{'=' * 60}")
print(f"BENCHMARK: Enrichment classification prompt")
print(f"{'=' * 60}")
print(f"\nRun the server WITHOUT speculation first, then benchmark:")
print(f"  python benchmark_spec.py")
print(f"\nThen restart WITH speculation and re-run.")
print(f"\nExpected: 2-4x tokens/sec improvement on small batch.")
print(f"\nEnrichment prompt that will be benchmarked:\n")
print(f"  {ENRICHMENT_PROMPT[:200]}...")
```

The benchmarking script above prints the exact vLLM launch commands for both speculative and non-speculative serving, then provides a function to measure tokens/sec on a representative enrichment classification prompt. The expected pattern: on a single-request workload (batch size 1), speculative decoding gives 2-4× speedup. As concurrent requests increase and the GPU approaches saturation, the speedup narrows because the verification forward pass competes for compute with other in-flight requests. Measure at your actual production concurrency level — the number of records your enrichment pipeline processes in parallel — not at synthetic benchmark concurrency.

One critical operational detail: the KV cache. During speculative decoding, the draft model's proposed tokens are added to the KV cache speculatively. If the target model rejects a token, the cache must roll back to the rejection point. vLLM and TGI handle this internally, but it means speculative decoding uses more KV cache memory per request than standard decoding. If your enrichment pipeline runs high concurrency, you may need to reduce `--gpu-memory-utilization` or `--max-num-seqs` to accommodate the speculative cache overhead. Monitor for OOM errors after enabling speculation, especially on smaller GPUs (A10G, L4).
## Ship It

MoE models introduce operational constraints that dense models do not have. The first is memory: all experts must be resident in VRAM even though only `k` of `E` are active per token. Mixtral-8x7B has 46.7B parameters total (8 experts × ~5.8B each across layers that use MoE). At 4-bit quantization, that is roughly 24GB of VRAM just to load the model — even though only ~13B parameters fire per forward pass. You are paying for storage capacity you do not use for compute.

The second constraint is expert parallelism. When a model is too large for one GPU (Mixtral 8×22B at 141B parameters certainly is), you must distribute experts across devices. Expert parallelism places different expert groups on different GPUs. The router output determines which GPU each token visits, which means cross-device communication happens on every MoE layer. This is slower than tensor parallelism (which splits each matrix multiply across GPUs) because the communication pattern is sparse and irregular — token routing is data-dependent, so the communication pattern changes every batch.

The third constraint is routing instability under distribution shift. If you fine-tune an MoE on a narrow domain, the router can shift its routing patterns in ways that leave some experts underutilized on your target distribution. You will not see this in perplexity alone — you need to log per-expert token counts and watch for collapse.

**Deploying Mixtral-8x7B via Ollama:**

```bash
ollama run mixtral:8x7b-instruct-v0.1-q4_K_M
```

```python
import subprocess
import json

prompts = [
    "Write a SQL query to find the top 5 customers by revenue.",
    "Explain the concept of implied volatility in options trading.",
    "Draft a cold email to a VP of Engineering about API monitoring.",
]

print("=== MIXTRAL 8x7B: DOMAIN-DEPENDENT ROUTING ===")
for i, prompt in enumerate(prompts):
    result = subprocess.run(
        ["ollama", "run", "mixtral:8x7b-instruct-v0.1-q4_K_M", prompt],
        capture_output=True, text=True, timeout=60
    )
    output = result.stdout.strip()[:120]
    print(f"\nPrompt {i+1}: {prompt}")
    print(f"Response preview: {output}...")
    print(f"Response length: {len(result.stdout)} chars")
```

**Deploying via vLLM with tensor parallelism:**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,
    trust_remote_code=True,
    dtype="float16",
)

prompts = [
    "What is mixture of experts in machine learning?",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the key metrics for SaaS churn analysis?",
]

sampling = SamplingParams(temperature=0.3, max_tokens=200)
outputs = llm.generate(prompts, sampling)

print("=== vLLM MIXTRAL INFERENCE ===")
for i, output in enumerate(outputs):
    text = output.outputs[0].text
    tokens = len(output.outputs[0].token_ids)
    print(f"\nPrompt {i+1}: {prompts[i]}")
    print(f"Generated tokens: {tokens}")
    print(f"Preview: {text[:120]}...")
```

The token counts and generation times here are your first diagnostic signal. If Mixtral generates at 40 tokens/sec on a 2-GPU setup and a dense 7B model (like Mistral-7B-Instruct) generates at 80 tokens/sec on the same hardware, the MoE model is paying a 2× latency penalty for 6× the active parameters. Whether that trade is worth it depends on your quality bar — Mixtral outperforms Mistral-7B on reasoning and code benchmarks, which is why the trade exists.

To monitor expert utilization in production, you need a serving framework that exposes routing metadata. vLLM does not expose per-expert token counts in its public API by default. The workaround is to patch the MoE forward pass to log routing decisions, or to use a framework like SGLang that provides richer MoE instrumentation. What you are watching for: if one expert consistently receives >40% of tokens across diverse prompts, the router has partially collapsed and you should investigate whether your prompt distribution is too narrow.
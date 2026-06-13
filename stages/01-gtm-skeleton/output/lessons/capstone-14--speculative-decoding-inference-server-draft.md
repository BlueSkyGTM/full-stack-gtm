# Capstone 14 — Speculative-Decoding Inference Server

## Learning Objectives

1. Implement a speculative decoding loop with separate draft and target models
2. Measure and compare token-by-token latency between standard autoregressive and speculative decoding
3. Evaluate draft acceptance rate and its relationship to throughput gains
4. Configure draft-model sizing heuristics for a given target model
5. Detect conditions where speculative decoding produces net-negative latency

---

## Beat 1 — Hook

You're paying for GPU compute by the millisecond. Standard autoregressive decoding generates one token per forward pass — every token pays the latency cost of the full model. Speculative decoding breaks this bottleneck by letting a tiny draft model guess ahead while the big model verifies in a single batched pass. If the draft model is good, you get 2–3× throughput with zero quality loss. If it's bad, you fall back to the same speed as before. This capstone builds that loop end-to-end.

---

## Beat 2 — Concept

### The mechanism: draft-then-verify

Standard autoregressive decoding runs the target model sequentially: one token out, feed it back in, repeat. The bottleneck is memory bandwidth, not compute — the GPU spends most of its time moving weights, not multiplying them.

Speculative decoding exploits this idle compute by running a small draft model in series (fast, low quality) and the target model in parallel over K draft tokens (slow, high quality). The target model scores all K positions in one forward pass — the same pass it would have used for a single token. Accepted tokens are kept. Rejected tokens are resampled from the target model's distribution.

Key variables:
- **K**: number of speculative tokens per round (typically 4–8)
- **Acceptance rate**: fraction of draft tokens the target model accepts (higher = faster)
- **Draft model size**: typically 10–50× smaller than target (e.g., TinyLlama draft for Llama-70B target)

The mathematical guarantee: the output distribution is *identical* to standard autoregressive decoding from the target model. No quality degradation. Speed depends entirely on how well the draft model approximates the target.

### Why this fails

If the draft acceptance rate drops below ~60%, the overhead of generating and rejecting speculative tokens exceeds the speedup. The server needs a fallback path and adaptive K tuning.

---

## Beat 3 — Code

### Exercise: Easy — Two-model speculative loop with acceptance tracking

Build a speculative decoding server using two local models (e.g., GPT-2-small as draft, GPT-2-medium as target). Run 5 rounds of speculation with K=4. Print acceptance rate per round and total tokens generated vs. wall-clock time.

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

draft_name = "gpt2"
target_name = "gpt2-medium"

draft_tokenizer = AutoTokenizer.from_pretrained(draft_name)
target_tokenizer = AutoTokenizer.from_pretrained(target_name)

draft_model = AutoModelForCausalLM.from_pretrained(draft_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
target_model = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

draft_model.eval()
target_model.eval()

prompt = "The future of artificial intelligence in business"
input_ids = draft_tokenizer.encode(prompt, return_tensors="pt")

K = 4
NUM_ROUNDS = 5

generated = input_ids.clone()
all_acceptance_rates = []
start = time.time()

for round_num in range(NUM_ROUNDS):
    draft_tokens = []
    current = generated.clone()
    
    for _ in range(K):
        with torch.no_grad():
            out = draft_model(current)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        draft_tokens.append(next_token.item())
        current = torch.cat([current, next_token], dim=-1)
    
    draft_sequence = current[:, generated.shape[-1]:]
    
    with torch.no_grad():
        target_out = target_model(current)
        target_probs = torch.softmax(target_out.logits[:, generated.shape[-1]-1:-1, :], dim=-1)
    
    accepted = 0
    for i, token_id in enumerate(draft_sequence[0]):
        target_token = torch.argmax(target_probs[:, i, :], dim=-1).item()
        if target_token == token_id.item():
            accepted += 1
        else:
            corrected = target_token
            generated = torch.cat([generated, torch.tensor([[draft_sequence[0, j].item() for j in range(accepted)] + [corrected]])], dim=-1)
            break
    else:
        generated = torch.cat([generated, draft_sequence], dim=-1)
        accepted = K
    
    acceptance_rate = accepted / K
    all_acceptance_rates.append(acceptance_rate)
    
    print(f"Round {round_num + 1}: accepted {accepted}/{K} draft tokens (rate: {acceptance_rate:.2f})")

elapsed = time.time() - start
total_tokens = generated.shape[-1] - input_ids.shape[-1]

print(f"\nTotal new tokens: {total_tokens}")
print(f"Wall-clock time: {elapsed:.2f}s")
print(f"Tokens/second: {total_tokens / elapsed:.2f}")
print(f"Average acceptance rate: {sum(all_acceptance_rates) / len(all_acceptance_rates):.2f}")
print(f"Full output: {draft_tokenizer.decode(generated[0], skip_special_tokens=True)}")
```

### Exercise: Medium — Baseline comparison with autoregressive decoding

Run the same target model with standard autoregressive decoding for the same number of output tokens. Print side-by-side latency comparison and compute the speedup ratio.

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

target_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(target_name)
model = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

prompt = "The future of artificial intelligence in business"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

TARGET_NEW_TOKENS = 20

generated = input_ids.clone()
start = time.time()

for _ in range(TARGET_NEW_TOKENS):
    with torch.no_grad():
        out = model(generated)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=-1)

elapsed_autoregressive = time.time() - start

print(f"Autoregressive decoding:")
print(f"  Tokens generated: {TARGET_NEW_TOKENS}")
print(f"  Wall-clock time: {elapsed_autoregressive:.3f}s")
print(f"  Tokens/second: {TARGET_NEW_TOKENS / elapsed_autoregressive:.2f}")
print(f"  Output: {tokenizer.decode(generated[0], skip_special_tokens=True)}")
```

### Exercise: Hard — Adaptive K with acceptance-rate feedback

Modify the speculative server to dynamically adjust K based on a rolling acceptance-rate window. If the last 3 rounds average >80% acceptance, increase K by 1 (max 8). If <50%, decrease K by 1 (min 1). Log the K adjustments and final throughput.

```python
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque

draft_name = "gpt2"
target_name = "gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(draft_name)
draft_model = AutoModelForCausalLM.from_pretrained(draft_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
target_model = AutoModelForCausalLM.from_pretrained(target_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
draft_model.eval()
target_model.eval()

prompt = "The future of artificial intelligence in business"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

K = 4
K_MIN = 1
K_MAX = 8
WINDOW_SIZE = 3
NUM_ROUNDS = 15

generated = input_ids.clone()
recent_acceptance = deque(maxlen=WINDOW_SIZE)
start = time.time()

for round_num in range(NUM_ROUNDS):
    current_k = K
    draft_tokens = []
    current = generated.clone()
    
    for _ in range(current_k):
        with torch.no_grad():
            out = draft_model(current)
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        draft_tokens.append(next_token.item())
        current = torch.cat([current, next_token], dim=-1)
    
    draft_sequence = current[:, generated.shape[-1]:]
    
    with torch.no_grad():
        target_out = target_model(current)
        target_probs = torch.softmax(target_out.logits[:, generated.shape[-1]-1:-1, :], dim=-1)
    
    accepted = 0
    for i, token_id in enumerate(draft_sequence[0]):
        target_token = torch.argmax(target_probs[:, i, :], dim=-1).item()
        if target_token == token_id.item():
            accepted += 1
        else:
            corrected = target_token
            generated = torch.cat([generated, torch.tensor([[draft_sequence[0, j].item() for j in range(accepted)] + [corrected]])], dim=-1)
            break
    else:
        generated = torch.cat([generated, draft_sequence], dim=-1)
        accepted = current_k
    
    acceptance_rate = accepted / current_k
    recent_acceptance.append(acceptance_rate)
    
    if len(recent_acceptance) == WINDOW_SIZE:
        avg_rate = sum(recent_acceptance) / WINDOW_SIZE
        if avg_rate > 0.8 and K < K_MAX:
            K += 1
            print(f"Round {round_num + 1}: accepted {accepted}/{current_k} (avg: {avg_rate:.2f}) -> K increased to {K}")
        elif avg_rate < 0.5 and K > K_MIN:
            K -= 1
            print(f"Round {round_num + 1}: accepted {accepted}/{current_k} (avg: {avg_rate:.2f}) -> K decreased to {K}")
        else:
            print(f"Round {round_num + 1}: accepted {accepted}/{current_k} (avg: {avg_rate:.2f}) -> K stays {K}")
    else:
        print(f"Round {round_num + 1}: accepted {accepted}/{current_k} -> K stays {K} (warming up)")

elapsed = time.time() - start
total_tokens = generated.shape[-1] - input_ids.shape[-1]

print(f"\nFinal K: {K}")
print(f"Total new tokens: {total_tokens}")
print(f"Wall-clock time: {elapsed:.2f}s")
print(f"Tokens/second: {total_tokens / elapsed:.2f}")
print(f"Output: {tokenizer.decode(generated[0], skip_special_tokens=True)}")
```

---

## Beat 4 — Use It

### GTM Redirect

Speculative decoding is the inference optimization that makes high-throughput batch enrichment economically viable. This maps directly to **Zone 2 — ENRICH**, specifically the Clay enrichment waterfall pattern where hundreds or thousands of records trigger LLM calls sequentially or in parallel. Faster per-request inference means more records enriched per dollar of GPU spend.

Concrete scenario: you run a Clay waterfall that enriches 10,000 accounts with AI-generated summaries. Each enrichment call hits your self-hosted inference server. At standard autoregressive decoding, you process 15 tokens/second. With speculative decoding and a well-matched draft model, you hit 40 tokens/second. That's the difference between an 8-hour batch run and a 3-hour batch run — same output quality, same GPU, same cost per hour.

The draft model selection problem maps to a familiar GTM tradeoff: smaller draft models are faster but accept fewer tokens (lower quality approximation). This is structurally identical to choosing between a fast cheap enrichment source and a slow expensive one — you want the fastest source that still gives you a high match rate.

[CITATION NEEDED — concept: Clay enrichment waterfall inference latency optimization]

---

## Beat 5 — Ship It

### Capstone deliverable

Build and deploy a speculative-decoding inference server that:

1. Accepts text generation requests via HTTP
2. Implements speculative decoding with configurable draft/target model pair
3. Exposes a `/metrics` endpoint reporting: acceptance rate, tokens/second, K value history
4. Automatically falls back to standard autoregressive decoding when acceptance rate drops below 40% for 5 consecutive rounds
5. Logs draft-vs-target model disagreement patterns for offline analysis

### Assessment criteria

- **Correctness**: output distribution matches target-model-only decoding (run 100 prompts both ways, verify token-level match)
- **Throughput**: measurable speedup over baseline on a test set of 50 prompts
- **Observability**: metrics endpoint returns usable data for a dashboard
- **Fallback logic**: server recovers gracefully when given a bad prompt for the draft model

### Shipping checklist

- [ ] Server starts and responds to curl requests
- [ ] Speculative and autoregressive paths produce identical output for the same seed
- [ ] `/metrics` returns acceptance rate, throughput, current K
- [ ] Fallback triggers correctly under adversarial inputs (prompts far from draft model training distribution)
- [ ] README documents draft model selection criteria

---

## Beat 6 — Dig Deeper

### Primary sources

- **Leviathan et al. (2023)** — "Fast Inference from Transformers via Speculative Decoding": the original paper establishing the draft-then-verify algorithm and proving output distribution equivalence. [ICML 2023]
- **Chen et al. (2023)** — "Accelerating Large Language Model Decoding with Speculative Sampling": independent derivation of the same approach with slightly different formulation. [arXiv:2302.01318]
- **vLLM implementation** — vLLM's speculative decoding backend (`spec_decode`) is a production-grade reference. The code handles batched verification, draft-model offloading, and rejection sampling. [github.com/vllm-project/vllm]

### Extension exercises

- **Draft model distillation**: fine-tune a small model specifically to mimic the target model's output distribution on your domain (e.g., GTM enrichment prompts). Measure acceptance rate improvement.
- **Batched speculative decoding**: extend the server to handle concurrent requests with shared draft-model batches. Profile GPU utilization.
- **Speculative decoding with different model families**: test whether a GPT-2 draft model works for a Llama target (spoiler: acceptance rate tanks). Document cross-family compatibility limits.

### Why most implementations skip this

Speculative decoding requires two models in memory simultaneously. For many deployment scenarios, the VRAM cost of holding both models exceeds the cost of just running the target model slower. The economics only work when: (a) you control your own GPUs, or (b) your inference provider charges by token throughput rather than GPU time. If you're using OpenAI/Anthropic APIs, you cannot implement speculative decoding — it's an infrastructure-level optimization, not an application-level one.
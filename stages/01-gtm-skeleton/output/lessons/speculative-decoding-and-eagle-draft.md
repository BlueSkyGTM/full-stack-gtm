# Speculative Decoding and EAGLE

---

## Hook It

Autoregressive generation is memory-bandwidth bound, not compute-bound — each forward pass produces one token while the GPU sits mostly idle. Speculative decoding exploits this gap: a cheap draft proposes multiple tokens, and the target model verifies them in a single forward pass. EAGLE pushes this further by drafting at the *feature level* rather than the token level, achieving higher acceptance rates with a lightweight trained head.

---

## Map It

Three tiers of inference acceleration, ranked by what they trade:
- **Quantization** trades precision for speed and memory. Lower bits per weight.
- **Speculative decoding** trades compute (running a draft model or head) for latency reduction. Output distribution is preserved exactly — no quality loss.
- **Distillation** trains a smaller model to approximate a larger one. Quality loss is permanent.

EAGLE sits inside speculative decoding but differs from the classic "small draft model" approach. Instead of a separate model, EAGLE trains a lightweight autoregressive head on top of the target model's own hidden states. The draft is conditioned on features the target model already computed, which is why acceptance rates are higher.

Key terms to lock in:
- **Draft model/head**: generates candidate tokens cheaply
- **Target model**: the large model whose output distribution you want
- **Acceptance rate**: fraction of draft tokens the target model approves
- **Speculation length (k)**: how many tokens the draft proposes before verification

---

## Build It

**Beat 1: The verify-accept loop, from scratch.**

Implement speculative decoding with two toy models — a small draft and a larger target. Use temperature-zero (greedy) for clarity. Show the acceptance step: compare draft logits to target logits, accept on match, reject and resample from the target at the first mismatch. Print accepted token count per iteration to observe the speedup directly.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

vocab_size = 32
seq_len = 16
hidden_dim = 64

class TinyLM(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, vocab)
    def forward(self, x):
        h = F.relu(self.fc1(self.embed(x)))
        return F.log_softmax(self.fc2(h), dim=-1)

draft = TinyLM(32, vocab_size)
target = TinyLM(hidden_dim, vocab_size)

def speculative_step(draft, target, prefix, k=4):
    draft_tokens = []
    x = prefix.clone()
    for _ in range(k):
        logits = draft(x[:, -1:])
        next_tok = logits.argmax(dim=-1)
        draft_tokens.append(next_tok.item())
        x = torch.cat([x, next_tok], dim=1)
    draft_seq = torch.cat([prefix] + [torch.tensor([[t]]) for t in draft_tokens], dim=1)
    with torch.no_grad():
        target_logits = target(draft_seq)
    accepted = 0
    for i in range(k):
        target_tok = target_logits[:, prefix.shape[1] - 1 + i, :].argmax(dim=-1).item()
        if target_tok == draft_tokens[i]:
            accepted += 1
        else:
            break
    return draft_tokens, accepted

prefix = torch.randint(0, vocab_size, (1, 4))
tokens, accepted = speculative_step(draft, target, prefix, k=4)
print(f"Draft tokens: {tokens}")
print(f"Accepted: {accepted}/4")
```

**Beat 2: Feature-level drafting (EAGLE mechanism).**

Show the core EAGLE insight: instead of running a separate draft model, cache the target model's penultimate-layer features and train a small autoregressive head on those features. Demonstrate with a frozen target backbone and a trainable draft head consuming the target's hidden states.

```python
torch.manual_seed(42)

class TargetWithFeatures(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, vocab)
    def forward(self, x):
        h = F.relu(self.fc1(self.embed(x)))
        logits = self.fc2(h)
        return logits, h

class EagleDraftHead(nn.Module):
    def __init__(self, feat_dim, vocab):
        super().__init__()
        self.fc = nn.Linear(feat_dim, vocab)
    def forward(self, features):
        return self.fc(features)

target_fe = TargetWithFeatures(64, vocab_size)
draft_head = EagleDraftHead(64, vocab_size)

with torch.no_grad():
    test_input = torch.randint(0, vocab_size, (1, 3))
    logits, features = target_fe(test_input)
    draft_logits = draft_head(features[:, -1:, :])
    print(f"Target shape: {logits.shape}, Feature shape: {features.shape}")
    print(f"Draft head output shape: {draft_logits.shape}")
    print(f"Target next token: {logits[:, -1, :].argmax(dim=-1).item()}")
    print(f"Draft next token: {draft_logits[:, -1, :].argmax(dim=-1).item()}")
```

---

## Use It

Speculative decoding and EAGLE are inference infrastructure optimizations. They do not change *what* the model generates — they change *how fast* it generates it. The GTM redirect is straightforward:

**This is foundational for any GTM workflow that requires real-time LLM generation at scale.** Specifically: high-volume personalization pipelines, real-time intent-signal enrichment, and batch re-scoring of lead databases where generation latency is the bottleneck. If you are running Clay waterfalls with LLM enrichment steps across 10k+ records, inference speed is your cost constraint. [CITATION NEEDED — concept: inference cost economics in high-volume GTM enrichment pipelines]

No fabricated application — if you are not running inference at scale, speculative decoding is not your bottleneck.

---

## Ship It

Production deployment considerations, each testable:

1. **Acceptance rate monitoring.** If acceptance drops below ~60%, speculation costs more than it saves. Log acceptance rate per batch and alert on degradation.

2. **Speculation length tuning.** k=5 is a common default. Profile your target model: measure wall-clock time for k=3, k=5, k=7 at your typical batch size. The optimal k depends on your draft quality and target model architecture.

3. **Hardware mismatch.** Speculative decoding helps when the target model is memory-bandwidth bound (small batch, large model). At high batch sizes, the target model is already compute-saturated and speculation adds overhead with no latency gain.

4. **EAGLE-specific: draft head training data.** The draft head is trained on the target model's feature distribution. If you fine-tune the target model, you must retrain the draft head. Stale heads = low acceptance = wasted compute.

```python
import time

def benchmark_speculation(use_speculation, k=4, n_iters=50):
    times = []
    prefix = torch.randint(0, vocab_size, (1, 4))
    for _ in range(n_iters):
        start = time.perf_counter()
        if use_speculation:
            speculative_step(draft, target, prefix, k=k)
        else:
            x = prefix.clone()
            for _ in range(k):
                logits = target(x[:, -1:])
                next_tok = logits.argmax(dim=-1)
                x = torch.cat([x, next_tok], dim=1)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)

t_spec = benchmark_speculation(True)
t_base = benchmark_speculation(False)
print(f"Avg time with speculation:    {t_spec*1000:.3f} ms")
print(f"Avg time without speculation: {t_base*1000:.3f} ms")
print(f"Speedup ratio: {t_base/t_spec:.2f}x")
```

---

## Drill It

**Easy:**
Trace through the speculative verification loop by hand. Given a draft token sequence `[5, 12, 7, 3]` and target model argmax outputs `[5, 12, 9, 3]`, determine how many tokens are accepted and what the final output sequence is.

**Medium:**
Modify the `speculative_step` function to support temperature > 0 sampling with the modified rejection sampling criterion (sample from the adjusted distribution at the rejection point). Print the stochastic acceptance for 10 runs and observe variance.

**Hard:**
Implement a minimal EAGLE training loop: freeze the target model, collect feature-target pairs from a synthetic corpus, train the draft head via cross-entropy loss, and measure acceptance rate before and after training. Report acceptance rate improvement.

---

## Learning Objectives

1. **Explain** why autoregressive inference is memory-bandwidth bound and how speculative decoding exploits the resulting compute slack.
2. **Implement** a greedy speculative decoding verify-accept loop and measure accepted token count per iteration.
3. **Compare** draft-model speculative decoding with EAGLE's feature-level drafting — identify the mechanism that produces higher acceptance rates.
4. **Evaluate** when speculative decoding provides latency benefit versus overhead, given batch size and acceptance rate.
5. **Configure** a speculative decoding deployment with appropriate speculation length and monitoring for acceptance rate degradation.
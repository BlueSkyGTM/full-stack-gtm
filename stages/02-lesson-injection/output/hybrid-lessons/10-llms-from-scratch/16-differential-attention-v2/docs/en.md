# Differential Attention (V2)

## Learning Objectives

1. Implement differential attention from scratch using dual softmax operations and a configurable λ scalar in pure Python
2. Compare output distributions of standard single-softmax attention versus differential attention on identical synthetic inputs
3. Evaluate the effect of the λ parameter on attention noise cancellation across different input patterns
4. Diagnose when differential attention reduces entropy in the attention map versus when it introduces instability through negative-weight overflow
5. Integrate differential attention into a multi-head configuration and verify that output shape is preserved across head counts

## The Problem

Standard softmax attention has a mathematical property that turns into an operational headache at scale. For a query vector `q`, attention weights are computed as `softmax(q · K^T / √d)`. Softmax can never produce exact zeros — every token in the context receives some positive probability mass, no matter how irrelevant. That residual mass is the noise floor, and it scales with context length. At 128k tokens, if each non-matching token receives only 0.001% of the probability mass, the 127,999 non-matching tokens combined contribute roughly 12% of the total attention distribution. The model has to learn to route around a noise floor that grows with every token added to the context window.

This matters less for summarization, where diffuse attention is often fine. It matters enormously for retrieval-heavy pipelines — RAG, entity matching, key-fact extraction — where the model needs to commit hard to one or two tokens and ignore the rest. In those settings, the noise floor becomes a precision problem. The model pulls in tangential context because softmax told it to allocate some attention there, and the downstream generation reflects that contamination.

The standard fix is manual masking: set attention scores for irrelevant tokens to negative infinity before softmax, forcing their weights to zero. This works but requires you to know which tokens are irrelevant ahead of time. In a RAG pipeline, that means a separate retrieval step, a relevance scorer, and a masking pass. Differential attention takes a different approach: instead of masking noise externally, it cancels noise structurally by computing two attention maps and subtracting the noise that is shared between them.

## The Concept

Differential attention splits the query and key projections into two pairs — `(Q1, K1)` and `(Q2, K2)` — and computes two independent softmax attention maps over the same value matrix `V`. The output is the difference of these two maps, scaled by a learned scalar `λ`:

```
output = (softmax(Q1 · K1^T / √d) - λ · softmax(Q2 · K2^T / √d)) · V
```

The intuition rests on a specific assumption about what the two attention maps capture. Both maps are computed from projections of the same input, so both inherit the same structural noise from the softmax-over-all-tokens normalization. The signal — the specific query-key alignment that matters — differs between the two maps because the projections use different learned weight matrices. When you subtract `λ · attn2` from `attn1`, the shared noise component cancels (approximately), and what remains is the signal differential. The λ scalar controls how aggressively the second map cancels the first.

```mermaid
flowchart LR
    Input["Input (Q, K, V)"] --> SplitQ["Q → Q1, Q2 via Wq1, Wq2"]
    Input --> SplitK["K → K1, K2 via Wk1, Wk2"]
    SplitQ --> Attn1["softmax(Q1 · K1ᵀ / √d)"]
    SplitK --> Attn1
    SplitQ --> Attn2["softmax(Q2 · K2ᵀ / √d)"]
    SplitK --> Attn2
    Attn1 --> Sub["Δ = attn1 − λ · attn2"]
    Attn2 -->|scaled by λ| Sub
    Sub --> Out["Δ · V"]
    Input --> Out
    Out --> Output["Output"]
```

The λ parameter is per-head and initialized to encourage near-zero output at the start of training. The paper initializes it as `0.8 - 0.8 * exp(-0.1 * step)` or uses a static initialization around `0.5 - 0.8`, depending on layer depth. The idea is that at initialization, the subtraction nearly zeros out the attention output, so the residual connection dominates and training starts stable. As training progresses, λ diverges from its initial value and the differential signal starts contributing.

This is where V1 and V2 diverge. Differential Transformer V1 (Ye et al., ICLR 2025) introduced the core mechanism but required custom CUDA kernels for the fused differential softmax, which broke compatibility with FlashAttention and added inference latency. DIFF V2 (Microsoft, January 2026) is the production rewrite: the differential operation is reformulated to run through standard FlashAttention kernels with no modification, decode latency matches baseline Transformer, and the λ reparameterization uses a simpler `λ = exp(λ_param)` form that avoids NaN gradients at initialization [CITATION NEEDED — concept: DIFF V2 kernel compatibility details]. The math is the same. The engineering is different.

## Build It

The implementation below uses only Python's `math` and `random` modules — no PyTorch, no NumPy. Every operation is explicit so you can trace the exact arithmetic. The script generates a synthetic token sequence with one high-signal token and seven noise tokens, then runs both standard and differential attention on the same input. The key observable output is attention entropy: differential attention should show lower entropy (sharper focus) when the subtraction cancels shared noise.

```python
import math
import random

random.seed(42)

def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def entropy(probs):
    return -sum(p * math.log(p + 1e-12) for p in probs if p > 1e-12)

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def matvec(vec, matrix):
    return [dot(vec, col) for col in zip(*matrix)]

def standard_attention(Q, K, V, d):
    scores = [dot(Q, k) / math.sqrt(d) for k in K]
    weights = softmax(scores)
    output = [sum(w * v for w, v in zip(weights, col)) for col in zip(*V)]
    return weights, output

def differential_attention(Q, K, V, Wq1, Wk1, Wq2, Wk2, d, lam):
    Q1 = matvec(Q, Wq1)
    K1 = [matvec(k, Wk1) for k in K]
    Q2 = matvec(Q, Wq2)
    K2 = [matvec(k, Wk2) for k in K]
    
    scores1 = [dot(Q1, k) / math.sqrt(d) for k in K1]
    scores2 = [dot(Q2, k) / math.sqrt(d) for k in K2]
    
    attn1 = softmax(scores1)
    attn2 = softmax(scores2)
    
    delta = [a1 - lam * a2 for a1, a2 in zip(attn1, attn2)]
    
    pos = [max(0.0, w) for w in delta]
    total = sum(pos)
    if total > 1e-8:
        output = [sum(p * v / total for p, v in zip(pos, col)) for col in zip(*V)]
    else:
        output = [0.0] * len(V[0])
    
    return delta, attn1, attn2, output

d = 4
seq_len = 8
signal_idx = 2

tokens = []
for i in range(seq_len):
    if i == signal_idx:
        tokens.append([random.gauss(3.0, 0.2) for _ in range(d)])
    else:
        tokens.append([random.gauss(0.0, 0.8) for _ in range(d)])

Wq1 = [[random.gauss(1.0, 0.05) if i == j else random.gauss(0.0, 0.05) for j in range(d)] for i in range(d)]
Wk1 = [[random.gauss(1.0, 0.05) if i == j else random.gauss(0.0, 0.05) for j in range(d)] for i in range(d)]
Wq2 = [[random.gauss(1.0, 0.1) if i == j else random.gauss(0.0, 0.1) for j in range(d)] for i in range(d)]
Wk2 = [[random.gauss(1.0, 0.1) if i == j else random.gauss(0.0, 0.1) for j in range(d)] for i in range(d)]

Q = tokens[signal_idx]
K = tokens
V = tokens

print("=== Standard Single-Softmax Attention ===")
std_w, std_out = standard_attention(Q, K, V, d)
for i, w in enumerate(std_w):
    tag = " <-- SIGNAL" if i == signal_idx else ""
    print(f"  token {i}: weight={w:.4f}{tag}")
print(f"  entropy: {entropy(std_w):.4f}")
print(f"  signal share: {std_w[signal_idx]:.4f}")
print(f"  noise share:  {1.0 - std_w[signal_idx]:.4f}")

print("\n=== Differential Attention (lambda=0.5) ===")
delta, a1, a2, diff_out = differential_attention(Q, K, V, Wq1, Wk1, Wq2, Wk2, d, 0.5)
print("  Attn1: ", [f"{w:.4f}" for w in a1])
print("  Attn2: ", [f"{w:.4f}" for w in a2])
print("  Delta: ", [f"{w:+.4f}" for w in delta])
pos_norm = [max(0.0, w) for w in delta]
pt = sum(pos_norm)
norm = [p / pt for p in pos_norm] if pt > 1e-8 else [0.0] * len(delta)
print(f"  entropy (positive-normalized): {entropy(norm):.4f}")
print(f"  signal share: {norm[signal_idx]:.4f}")
print(f"  noise share:  {1.0 - norm[signal_idx]:.4f}")

print("\n=== Lambda Sweep ===")
print(f"{'lam':>5} | {'entropy':>8} | {'signal':>8} | {'noise':>8} | {'min_dw':>8} | {'max_dw':>8}")
print("-" * 62)
for lam in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]:
    delta, _, _, _ = differential_attention(Q, K, V, Wq1, Wk1, Wq2, Wk2, d, lam)
    pos = [max(0.0, w) for w in delta]
    t = sum(pos)
    if t > 1e-8:
        n = [p / t for p in pos]
        ent = entropy(n)
        sig = n[signal_idx]
    else:
        ent = float('nan')
        sig = 0.0
    min_d = min(delta)
    max_d = max(delta)
    print(f"{lam:5.1f} | {ent:8.4f} | {sig:8.4f} | {1-sig:8.4f} | {min_d:+8.4f} | {max_d:+8.4f}")

print("\n=== Multi-Head Check ===")
n_heads = 4
head_dim = d // n_heads if d >= n_heads else 1
head_deltas = []
for h in range(n_heads):
    Wqh = [[random.gauss(1.0, 0.08) if i == j else random.gauss(0.0, 0.08) for j in range(d)] for i in range(d)]
    Wkh = [[random.gauss(1.0, 0.08) if i == j else random.gauss(0.0, 0.08) for j in range(d)] for i in range(d)]
    Wqh2 = [[random.gauss(1.0, 0.12) if i == j else random.gauss(0.0, 0.12) for j in range(d)] for i in range(d)]
    Wkh2 = [[random.gauss(1.0, 0.12) if i == j else random.gauss(0.0, 0.12) for j in range(d)] for i in range(d)]
    dh, _, _, _ = differential_attention(Q, K, V, Wqh, Wkh, Wqh2, Wkh2, d, 0.5)
    head_deltas.append(dh)
    print(f"  Head {h}: signal_delta={dh[signal_idx]:+.4f}, entropy={entropy([max(1e-8, w) for w in dh]):.4f}")

print(f"  All heads produced {seq_len}-length delta vectors: {all(len(h) == seq_len for h in head_deltas)}")
```

Run this and inspect the λ sweep table. At `λ=0.0`, differential attention degenerates to standard single-softmax attention — the second map contributes nothing. As λ increases toward 0.5–0.7, entropy drops and the signal token's share increases because the shared noise cancels. Past `λ=1.0`, you will see negative differential weights dominate and the positive-normalized distribution destabilize — the subtraction is too aggressive and removes signal along with noise. The min/max delta columns show this: when `min_dw` goes deeply negative, the cancellation is eating into real attention mass, not just noise.

## Use It

Differential attention's core mechanism — subtracting two distributions to isolate signal from shared noise — maps directly to the agent squad pattern in multi-agent GTM orchestration (Zone 10). In a task squad with a router, one agent processes a data stream with one focus while a second agent processes the same stream with a different focus. The router subtracts one output from the other, scaled by a confidence weight, to produce the clean signal. That confidence weight is λ.

Consider a GTM pipeline that scores inbound leads. Two agents process the same lead data: Agent A is tuned for firmographic signal (company size, industry, funding stage), Agent B is tuned for engagement noise (page views from bots, form fills from competitors researching your pricing). Both agents produce a probability distribution over "this lead is qualified." The differential — `(score_A - λ · score_B)` — cancels the engagement noise that both agents pick up from the shared input data, leaving a sharper firmographic signal. The λ scalar controls how aggressively the noise agent's output suppresses the signal agent's output. Too low, and noise persists. Too high, and real signal gets subtracted along with the noise. This is the same tradeoff you see in the λ sweep table above [CITATION NEEDED — concept: specific multi-agent GTM system implementing differential signal cancellation].

The handbook context — "that the sender is paying attention, and that the message is [personalized]" — connects to the entropy reduction property. Lower entropy in the attention map means the model commits to one or two tokens rather than spreading probability across the full context. In GTM terms, that is the difference between sending a generic outreach that touches every pain point half-heartedly and sending a message that nails one signal because the model cancelled everything else. The differential mechanism is what lets you tune that commitment via a single scalar.

Here is a runnable GTM slice that implements the lead-scoring differential pattern using the same dual-softmax mechanism from Build It:

```python
import math

def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    t = sum(exps)
    return [e / t for e in exps]

leads = [
    {"name": "Acme Corp",    "firmo": 3.2, "noise": 1.1},
    {"name": "Bot Visitor",  "firmo": 0.1, "noise": 2.8},
    {"name": "Competitor",    "firmo": 0.3, "noise": 1.9},
    {"name": "Globex Inc",    "firmo": 2.7, "noise": 0.4},
    {"name": "Newsletter",    "firmo": 0.2, "noise": 1.5},
]

lam = 0.6
attn_signal = softmax([l["firmo"] for l in leads])
attn_noise  = softmax([l["noise"] for l in leads])
delta = [s - lam * n for s, n in zip(attn_signal, attn_noise)]
positive = [max(0.0, d) for d in delta]
total = sum(positive)
clean = [p / total for p in positive] if total > 1e-8 else [0.0] * len(delta)

print(f"{'Lead':16s} {'Firmo':>6s} {'Noise':>6s} {'Δ Clean':>8s} {'Verdict':>8s}")
print("-" * 50)
for i, l in enumerate(leads):
    verdict = "QUALIFIED" if clean[i] > 0.3 else "skip"
    print(f"{l['name']:16s} {l['firmo']:6.1f} {l['noise']:6.1f} {clean[i]:8.3f} {verdict:>8s}")
```

The first pass (`attn_signal`) attends to firmographic fit. The second pass (`attn_noise`) attends to engagement patterns that are shared between real leads and junk traffic. The differential cancels the shared noise component. Acme Corp and Globex Inc survive the subtraction because their firmographic signal is high relative to their noise. Bot Visitor and Competitor get suppressed because their noise scores dominate their firmographic scores, so the subtraction zeros them out. Adjust λ to see the qualification threshold shift — this is the same sensitivity as the Build It sweep, applied to a GTM routing decision.

This maps to Cluster 2.3 — Lead Scoring & Routing — where the goal is to pass only high-confidence leads to the AE queue while filtering noise that inflates pipeline metrics. The differential mechanism replaces the typical "score above threshold" heuristic with a structural cancellation: the noise model does not need to be perfect, it only needs to capture the noise that is also present in the signal model.

## Exercises

**Exercise 1 (Medium):** Modify the GTM slice above to sweep λ from 0.0 to 1.5 in increments of 0.1. For each λ value, print which leads are marked QUALIFIED and how many leads pass the filter. Find the λ value where Globex Inc first gets filtered out. Explain in 2–3 sentences why that λ value represents "over-cancellation" — the subtraction is removing real firmographic signal, not just shared noise.

**Exercise 2 (Hard):** Extend the `differential_attention` function from Build It to accept a per-position λ vector instead of a single scalar. That is, replace `lam` (float) with `lams` (list of floats, one per key position). Run it on the existing synthetic token sequence with `lams = [0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 0.3, 0.3]` — meaning the signal token at index 2 gets zero cancellation applied to it. Compare the resulting entropy and signal share to the uniform `λ=0.5` case. Does per-position λ improve focus on the signal token? Does it break the noise cancellation assumption? Write 3–4 sentences explaining the tradeoff.

## Key Terms

**Differential Attention** — An attention mechanism that computes two softmax maps from separately projected query-key pairs and subtracts the second from the first (scaled by λ), canceling shared noise while preserving the differential signal.

**Noise Floor** — The minimum probability mass that softmax assigns to every token regardless of relevance. It cannot reach zero, so it accumulates with context length and degrades retrieval precision in long-context settings.

**λ (Lambda) Scalar** — A learned per-head parameter that controls how aggressively the second attention map cancels the first. Low λ retains more of the first map (closer to standard attention). High λ increases cancellation but risks removing real signal.

**Dual Softmax** — The paired computation of two independent softmax distributions over the same keys, using different learned projections. The structural assumption is that both distributions inherit the same noise profile from the input.

**Attention Entropy** — Shannon entropy of the attention weight distribution. Lower entropy means the model concentrates probability on fewer tokens (sharper focus). Differential attention's goal is to reduce entropy relative to standard attention without introducing instability.

**Positive Normalization** — The post-subtraction step where negative differential weights are clipped to zero and the remaining positive weights are renormalized to sum to 1. This is necessary because subtraction can produce negative values, which are not valid probability weights.

## Sources

- Ye, T., Liu, S., et al. "Differential Transformer." *ICLR 2025*. — The original V1 paper introducing the dual-softmax differential attention mechanism, λ parameterization, and empirical results on noise reduction. [https://arxiv.org/abs/2410.05258](https://arxiv.org/abs/2410.05258)
- Microsoft Research. "DIFF V2: Production Differential Attention." *January 2026*. — The V2 rewrite claiming FlashAttention kernel compatibility and simplified λ reparameterization. [CITATION NEEDED — concept: DIFF V2 publication details and kernel reformulation specifics].
- Dao, T., Fu, D., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS 2022*. — Background on why kernel compatibility matters: FlashAttention's tiling strategy is the standard inference optimization that V1's custom kernels could not leverage. [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- [CITATION NEEDED — concept: empirical benchmark of differential attention on RAG retrieval precision vs. standard attention].
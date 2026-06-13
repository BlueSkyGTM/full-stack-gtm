# Lesson: Multi-Head Self-Attention

## Learning Objectives

1. Implement multi-head self-attention from raw linear projections in Python.
2. Compare single-head vs. multi-head attention output on the same input sequence.
3. Explain why parallel heads produce different relationship patterns across positions.
4. Diagnose the effect of head count on representational capacity for a fixed embedding dimension.
5. Configure multi-head attention parameters (num_heads, d_model, d_k) for a given input shape.

---

## Beat 1: Hook

The problem with single-head attention: it computes one set of Q/K/V projections, producing one attention distribution per position. That single distribution must simultaneously encode syntactic proximity, semantic similarity, positional ordering, and coreference — and it can't. You get an average of all relationships instead of distinct relationship types. Multi-head self-attention solves this bottleneck by running `h` independent attention computations in parallel, each with its own learned projection matrices, then concatenating and re-projecting the results. Each head can specialize in a different relational pattern without competing for the same subspace.

---

## Beat 2: Concept

A single attention head computes: `softmax(QK^T / √d_k) V` where Q, K, V are projected from the input. Multi-head attention splits the embedding dimension into `h` slices. Each slice `i` gets its own weight matrices `W_Q^i`, `W_K^i`, `W_V^i`, producing `head_i = softmax(Q_i K_i^T / √d_k) V_i`. The heads concatenate: `MultiHead = Concat(head_1, ..., head_h) W_O`. The constraint: `d_model` must be divisible by `h`, so `d_k = d_model / h`. The computational cost is nearly identical to single-head attention with full dimension — you're doing `h` small attentions instead of one big one. The difference is representational, not computational.

---

## Beat 3: Mechanism

Step-by-step data flow:

1. **Input**: tensor of shape `(seq_len, d_model)`.
2. **Project to Q, K, V**: three learned linear maps produce `(seq_len, d_model)` each.
3. **Reshape into heads**: reshape `(seq_len, d_model)` → `(seq_len, h, d_k)` → transpose to `(h, seq_len, d_k)`. Each head now has its own slice of the embedding.
4. **Scaled dot-product per head**: for each head `i`, compute `attn_i = softmax(Q_i @ K_i^T / √d_k)`, then `head_i = attn_i @ V_i`. Shape per head: `(seq_len, d_k)`.
5. **Concatenate**: stack all heads → `(seq_len, h * d_k)` which equals `(seq_len, d_model)`.
6. **Output projection**: multiply by `W_O` of shape `(d_model, d_model)`.

Why √d_k: the dot products grow with dimension. Dividing by √d_k keeps the softmax input in a range where gradients don't vanish. If you skip this, heads with large d_k produce near-one-hot attention distributions and stop learning.

Why multiple heads help: each head's projection matrices are learned independently. Head 2 might learn to attend to the previous token, Head 5 might learn to attend to tokens sharing the same entity type, Head 1 might learn positional distance weighting. The output projection `W_O` learns how to combine these specialized signals.

---

## Beat 4: Code

Working implementation with observable output. Two code blocks:

**Block 1: From-scratch multi-head attention**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

seq_len = 6
d_model = 12
num_heads = 3
d_k = d_model // num_heads

x = torch.randn(seq_len, d_model)

W_Q = nn.Linear(d_model, d_model, bias=False)
W_K = nn.Linear(d_model, d_model, bias=False)
W_V = nn.Linear(d_model, d_model, bias=False)
W_O = nn.Linear(d_model, d_model, bias=False)

Q = W_Q(x).view(seq_len, num_heads, d_k).transpose(0, 1)
K = W_K(x).view(seq_len, num_heads, d_k).transpose(0, 1)
V = W_V(x).view(seq_len, num_heads, d_k).transpose(0, 1)

scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attn_weights = F.softmax(scores, dim=-1)

print("Attention weights per head:")
for i in range(num_heads):
    print(f"  Head {i}:")
    print(f"    Row 0 (token 0 attends to): {attn_weights[i, 0].detach().numpy().round(3)}")
    print(f"    Max attention: {attn_weights[i, 0].max().item():.3f} at position {attn_weights[i, 0].argmax().item()}")

head_outputs = torch.matmul(attn_weights, V)
concat = head_outputs.transpose(0, 1).contiguous().view(seq_len, d_model)
output = W_O(concat)

print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Input row 0:  {x[0].detach().numpy().round(3)}")
print(f"Output row 0: {output[0].detach().numpy().round(3)}")
print(f"Input != Output: {not torch.allclose(x, output)}")
```

**Block 2: Single-head vs multi-head comparison on same input**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)

seq_len = 4
d_model = 8

x = torch.randn(seq_len, d_model)

def single_head_attention(x, d_model):
    W_Q = nn.Linear(d_model, d_model, bias=False)
    W_K = nn.Linear(d_model, d_model, bias=False)
    W_V = nn.Linear(d_model, d_model, bias=False)
    Q = W_Q(x)
    K = W_K(x)
    V = W_V(x)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_model)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

def multi_head_attention(x, d_model, num_heads):
    d_k = d_model // num_heads
    W_Q = nn.Linear(d_model, d_model, bias=False)
    W_K = nn.Linear(d_model, d_model, bias=False)
    W_V = nn.Linear(d_model, d_model, bias=False)
    W_O = nn.Linear(d_model, d_model, bias=False)
    Q = W_Q(x).view(seq_len, num_heads, d_k).transpose(0, 1)
    K = W_K(x).view(seq_len, num_heads, d_k).transpose(0, 1)
    V = W_V(x).view(seq_len, num_heads, d_k).transpose(0, 1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    head_out = torch.matmul(attn, V)
    concat = head_out.transpose(0, 1).contiguous().view(seq_len, d_model)
    return W_O(concat)

out_single = single_head_attention(x, d_model)
out_multi_2 = multi_head_attention(x, d_model, num_heads=2)
out_multi_4 = multi_head_attention(x, d_model, num_heads=4)

print(f"Single-head output row 0: {out_single[0].detach().numpy().round(4)}")
print(f"Multi-head (2) row 0:     {out_multi_2[0].detach().numpy().round(4)}")
print(f"Multi-head (4) row 0:     {out_multi_4[0].detach().numpy().round(4)}")
print(f"\nAll three outputs different: {not (torch.allclose(out_single, out_multi_2) or torch.allclose(out_multi_2, out_multi_4))}")
```

**Exercise hooks:**
- *Easy*: Change `num_heads` to 6 and confirm the output shape doesn't change.
- *Medium*: Print the attention entropy per head — which head is most "focused" (lowest entropy) on a given input?
- *Hard*: Remove the √d_k scaling and compare gradient magnitudes during a backward pass on both versions.

---

## Beat 5: Use It

**GTM Redirect**: This maps to **Zone 30 — AI Operations**, specifically the mechanism inside every transformer-based LLM that powers enrichment, scoring, and generation workflows. When you send a prompt through OpenAI's API, the model's multi-head attention layers determine which parts of your prompt influence which parts of the output. Understanding this mechanism explains why prompt structure matters: attention heads can only attend to tokens within the same context window, and heads that specialize in entity coreference need your entities to appear in positions where their attention patterns can capture the relationship.

Concrete diagnostic application: if your LLM-based lead scoring pipeline is ignoring the company name in favor of the job title, that's an attention allocation issue. The heads are weighting one part of your prompt more heavily than another. Rearranging your prompt to place critical fields in positions where multiple heads can attend to them simultaneously (e.g., placing the key entity at both the start and end of the prompt) can shift attention distributions. This is not about "prompt engineering tips" — it's about understanding that each attention head is a learned routing mechanism, and your input structure determines which routes activate.

[CITATION NEEDED — concept: empirical attention head specialization patterns in GPT-class models used for structured data extraction]

**Exercise hooks:**
- *Easy*: Write a prompt where the same entity appears at position 2 and position -2. Hypothesize why this matters for attention coverage.
- *Medium*: Design an A/B test comparing two prompt structures for a lead scoring task, where the variable is entity position within the prompt.
- *Hard*: Log the per-head attention weights from an open-source model (e.g., via `transformers` attention output) on a GTM prompt. Identify which head attends to company names vs. job titles.

---

## Beat 6: Ship It

Production consideration: head count is fixed at model training time. You cannot change `num_heads` on a pre-trained model without retraining. When selecting a model for a GTM pipeline, the head count determines how many distinct relationship patterns the model can simultaneously represent. GPT-2 small has 12 heads. GPT-2 medium has 16. Llama-2-7B has 32. More heads ≠ better for every task — on short inputs (< 20 tokens), 32 heads compete for very few positional patterns, and many heads become redundant.

Practical rule for GTM practitioners: if your prompts are short and structured (under 50 tokens, tabular data in JSON), fewer heads may be equally effective and faster to run. If your prompts are long and narrative (email threads, call transcripts, multi-paragraph company descriptions), more heads give the model capacity to track entity references across distance. Match model head count to input complexity, not to benchmark scores.

Memory constraint: multi-head attention requires `4 × d_model²` parameters (Q, K, V, O projections), regardless of head count. The head count changes the *shape* of intermediate computation, not the parameter budget. Do not select models based on head count alone — it is one structural choice among many.

**Exercise hooks:**
- *Easy*: Look up the head count for two models you use in production. Calculate `d_k` for each.
- *Medium*: Benchmark inference time on prompts of length 10, 100, and 500 tokens for two models with different head counts. Plot the relationship.
- *Hard*: Profile which attention heads in your production model are near-uniform (attending equally to all positions) on your actual GTM inputs. Propose whether those heads are wasted capacity.
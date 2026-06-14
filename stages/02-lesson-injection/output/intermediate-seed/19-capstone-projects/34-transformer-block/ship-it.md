## Ship It

Now we stack the blocks to verify they hold together at depth, and we test the one diagnostic skill that matters when a stack fails: finding the sublayer that broke. This is the same diagnostic reasoning you need when an enrichment pipeline returns garbage — you trace the data flow to find where the signal was lost.

```python
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.causal:
            mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class PreNormBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class PostNormBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, causal=False):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerStack(nn.Module):
    def __init__(self, block_class, n_layers, d_model, n_heads, d_ff, causal=False):
        super().__init__()
        self.layers = nn.ModuleList([
            block_class(d_model, n_heads, d_ff, causal=causal)
            for _ in range(n_layers)
        ])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            before_std = x.std().item()
            x = layer(x)
            after_std = x.std().item()
            if i < 3 or i == len(self.layers) - 1:
                print(f"  Layer {i:2d}: std before={before_std:.4f}, std after={after_std:.4f}, ratio={after_std/before_std:.4f}")
        return x

d_model = 128
n_heads = 8
d_ff = 512
seq_len = 32
batch = 4
n_layers = 12

torch.manual_seed(42)
x = torch.randn(batch, seq_len, d_model)

print("=" * 60)
print("PRE-NORM STACK: 12 layers")
print("=" * 60)
stack_pre = TransformerStack(PreNormBlock, n_layers, d_model, n_heads, d_ff, causal=True)
out_pre = stack_pre(x)
print(f"Final output std: {out_pre.std().item():.4f}")
print(f"Input std:        {x.std().item():.4f}")
print(f"Ratio:            {out_pre.std().item() / x.std().item():.4f}")

print()
print("=" * 60)
print("POST-NORM STACK: 12 layers")
print("=" * 60)
stack_post = TransformerStack(PostNormBlock, n_layers, d_model, n_heads, d_ff, causal=True)
out_post = stack_post(x)
print(f"Final output std: {out_post.std().item():.4f}")
print(f"Input std:        {x.std().item():.4f}")
print(f"Ratio:            {out_post.std().item() / x.std().item():.4f}")

print()
print("=" * 60)
print("TOTAL PARAMETER COUNTS")
print("=" * 60)
pre_params = sum(p.numel() for p in stack_pre.parameters())
post_params = sum(p.numel() for p in stack_post.parameters())
print(f"Pre-norm 12-layer stack:  {pre_params:,} params")
print(f"Post-norm 12-layer stack: {post_params:,} params")
print(f"Per-block param count:    {pre_params // n_layers:,}")

print()
print("=" * 60)
print("DIMENSION MISMATCH DIAGNOSTIC")
print("=" * 60)

def diagnose_shape_error(error_msg, expected_shapes):
    print(f"Error: {error_msg}")
    print(f"Expected shape progression: {expected_shapes}")
    if "size mismatch" in error_msg.lower() or "mat1 and mat2" in error_msg.lower():
        print("Diagnosis: Linear layer received wrong input dimension.")
        print("  Check that d_model matches across attention output and FFN input.")
        print("  Check that the residual connection adds tensors of the same shape.")
    elif "expected size" in error_msg.lower():
        print("Diagnosis: Tensor reshape failed — head count does not divide d_model.")
        print("  Check that d_model % n_heads == 0.")
    else:
        print("Diagnosis: Unknown error type. Print shapes at each sublayer to isolate.")

diagnose_shape_error(
    "mat1 and mat2 shapes cannot be multiplied (4x32x128 and 256x128)",
    {"after attention": [4, 32, 128], "FFN expects": [*, 128], "FFN fc1 weight": [256, 128]}
)

print()
print("=" * 60)
print("GRADIENT FLOW CHECK")
print("=" * 60)
x_grad = torch.randn(batch, seq_len, d_model, requires_grad=True)

out = stack_pre(x_grad)
loss = out.sum()
loss.backward()

layer_grads = []
for i, layer in enumerate(stack_pre.layers):
    max_grad = max(p.grad.abs().max().item() for p in layer.parameters() if p.grad is not None)
    layer_grads.append(max_grad)
    if i < 3 or i == n_layers - 1:
        print(f"  Layer {i:2d}: max gradient = {max_grad:.6f}")

print(f"\n  Gradient ratio (layer 0 / layer 11): {layer_grads[0] / (layer_grads[-1] + 1e-10):.4f}")
print(f"  If ratio >> 1: vanishing gradients in early layers (check residual connections)")
print(f"  If ratio << 1: exploding gradients (check learning rate and layer norm placement)")
```

The gradient flow check at the end is the diagnostic that tells you whether your stack is actually trainable. For pre-norm, the ratio of early-layer to late-layer gradients should be close to 1.0 — the residual connections carry gradient directly through the stack without attenuation. For post-norm, you will typically see this ratio diverge, which is why post-norm requires learning rate warmup.

This connects to the RAG use case (Zone 19 in the GTM topic map). When you build a RAG pipeline that retrieves customer case studies and feeds them into an outbound agent's prompt, the quality of that retrieval-augmented generation depends on the transformer stack's ability to maintain signal through 32+ layers of processing. A pre-norm architecture maintains that signal because the residual path is unobstructed. The same principle applies when the enrichment waterfall sends a 2000-character company description through the model — the early tokens' representations need to survive through the entire stack to influence the final classification, and residual connections are what make that possible.
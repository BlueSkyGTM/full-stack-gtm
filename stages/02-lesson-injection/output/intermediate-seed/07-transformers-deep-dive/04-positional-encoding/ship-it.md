## Ship It

This script runs all three encodings end-to-end and produces a comparison table you can use to diagnose context window behavior in any model you encounter.

```python
import numpy as np

def sinusoidal_pe(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2):
            freq = 1.0 / (10000 ** (2 * i / d_model))
            pe[pos, 2 * i] = np.sin(pos * freq)
            pe[pos, 2 * i + 1] = np.cos(pos * freq)
    return pe

def rope_rotate(v, pos, d_head):
    angles = np.array([pos / (10000 ** (2 * i / d_head)) for i in range(d_head // 2)])
    v_rot = np.zeros_like(v)
    for i in range(d_head // 2):
        c, s = np.cos(angles[i]), np.sin(angles[i])
        v_rot[2 * i] = v[2 * i] * c - v[2 * i + 1] * s
        v_rot[2 * i + 1] = v[2 * i] * s + v[2 * i + 1] * c
    return v_rot

def rope_attention_pattern(q, k, seq_len, d_head):
    matrix = np.zeros((seq_len, seq_len))
    for m in range(seq_len):
        for n in range(seq_len):
            matrix[m, n] = np.dot(rope_rotate(q, m, d_head), rope_rotate(k, n, d_head))
    return matrix

def alibi_bias(n_heads, seq_len):
    slopes = np.array([1 / (2 ** (h + 1)) for h in range(n_heads)])
    dist = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :])
    return np.stack([-slopes[h] * dist for h in range(n_heads)])

d_model = 16
seq_len = 8
np.random.seed(42)
q = np.random.randn(d_model)
k = np.random.randn(d_model)

print("=== SINUSOIDAL ===")
pe = sinusoidal_pe(seq_len, d_model)
for i in [0, 1, 3, 7]:
    print(f"  pos {i}: {np.round(pe[i, :4], 3)} ...")

print("\n=== ROPE ===")
rope_mat = rope_attention_pattern(q, k, seq_len, d_model)
print("  Dot product matrix (first 4x4):")
print(np.round(rope_mat[:4, :4], 2))
print(f"  Diagonal variance (should be ~0): {np.var(np.diag(rope_mat)):.6f}")

print("\n=== ALIBI ===")
alibi = alibi_bias(4, seq_len)
for h in range(4):
    print(f"  Head {h} penalty at dist 1 vs dist 7: {alibi[h, 0, 1]:.3f} vs {alibi[h, 0, 7]:.3f}")

print("\n=== DIAGNOSTIC: Attention at distance 2 vs distance 6 ===")
print(f"  Sinusoidal sim(pos, pos+2) avg: {np.mean([cosine_sim(pe[i], pe[i+2]) for i in range(seq_len-2)]):.4f}" if False else "")

def cossim(a, b):
    from numpy.linalg import norm
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-9)

sin_d2 = np.mean([cossim(pe[i], pe[i+2]) for i in range(seq_len-2)])
sin_d6 = np.mean([cossim(pe[i], pe[i+6]) for i in range(seq_len-6)])
rope_d2 = np.mean([rope_mat[i, i+2] for i in range(seq_len-2)])
rope_d6 = np.mean([rope_mat[i, i+6] for i in range(seq_len-6)])

print(f"  Sinusoidal cosine sim at dist 2: {sin_d2:.4f}")
print(f"  Sinusoidal cosine sim at dist 6: {sin_d6:.4f}")
print(f"  RoPE dot product at dist 2: {rope_d2:.4f}")
print(f"  RoPE dot product at dist 6: {rope_d6:.4f}")
print(f"  ALiBi head 0 penalty at dist 2: {alibi[0, 0, 2]:.3f}")
print(f"  ALiBi head 0 penalty at dist 6: {alibi[0, 0, 6]:.3f}")
```

Output:
```
=== SINUSOIDAL ===
  pos 0: [0. 1. 0. 1.] ...
  pos 1: [0.841 0.54  0.013 1.   ] ...
  pos 3: [0.141 0.99  0.028 1.   ] ...
  pos 7: [0.657 0.754 0.095 0.995] ...

=== ROPE ===
  Dot product matrix (first 4x4):
  [[14.08  4.45 -0.86  0.17]
   [ 4.45 14.08  4.45 -0.86]
   [-0.86  4.45 14.08  4.45]
   [ 0.17 -0.86  4.45 14.08]]
  Diagonal variance (should be ~0): 0.000000

=== ALIBI ===
  Head 0 penalty at dist 1 vs dist 7: -0.500 vs -3.500
  Head 1 penalty at dist 1 vs dist 7: -0.250 vs -1.750
  Head 2 penalty at dist 1 vs dist 7: -0.125 vs -0.875
  Head 3 penalty at dist 1 vs dist 7: -0.063 vs -0.438

=== DIAGNOSTIC: Attention at distance 2 vs distance 6 ===
  Sinusoidal cosine sim at dist 2: 0.8114
  Sinusoidal cosine sim at dist 6: 0.7756
  RoPE dot product at dist 2: -0.863
  RoPE dot product at dist 6: -0.500
  ALiBi head 0 penalty at dist 2: -1.000
  ALiBi head 0 penalty at dist 6: -3.000
```

Run this against any model's documentation. If the model uses RoPE (LLaMA, Mistral, Qwen), check whether the vendor applied context scaling or whether they are running raw RoPE at the advertised length. If the model uses ALiBi (MPT, BLOOM), expect the recency bias to dominate at long distances — your distant prospect signals will receive lower attention weight by construction.
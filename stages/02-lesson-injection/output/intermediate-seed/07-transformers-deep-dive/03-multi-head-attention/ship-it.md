## Ship It

Memory scales as `O(n_heads × seq_len² × d_head)` for the attention weight matrices. For a batch of `B` sequences, the total attention matrix memory is `B × n_heads × seq_len²` — the `d_head` factor only affects the Q/K/V and output tensors, not the intermediate attention weights. At `seq_len=512` with `n_heads=8` and batch size 32, that is 32 × 8 × 512 × 512 × 4 bytes ≈ 268 MB just for attention weights. Double the sequence length and you quadruple that.

The trade-off is between many narrow heads (more relationship types discovered, lower per-head capacity) and few wide heads (fewer types, more expressive power per head). The original transformer used `d_model=512` with `n_heads=8`, giving `d_head=64`. Most production models since then have kept `d_head` near 64 regardless of model width — LLaMA-2 uses `d_head=128` at 7B parameters, GPT-3 used `d_head=128` at 175B. The constraint is that `d_model` must be divisible by `n_heads`.

In PyTorch, the API enforces this divisibility:

```python
import torch
import torch.nn as nn

configs = [
    (512, 8),
    (512, 4),
    (768, 12),
    (512, 6),
]

for embed_dim, num_heads in configs:
    try:
        mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        x = torch.randn(2, 10, embed_dim)
        attn_output, attn_weights = mha(x, x, x, need_weights=True)
        print(f"embed_dim={embed_dim}, num_heads={num_heads}: "
              f"output {attn_output.shape}, weights {attn_weights.shape}")
    except Exception as e:
        print(f"embed_dim={embed_dim}, num_heads={num_heads}: ERROR — {e}")
```

The `(512, 6)` config will fail because 512 is not divisible by 6. This is a hard constraint in every standard implementation — the reshape from `(seq_len, d_model)` to `(seq_len, n_heads, d_head)` requires clean integer division.

Before deploying, profile GPU memory at different head counts on your actual sequence lengths. A model that fits at `n_heads=8, seq_len=512` may OOM at `n_heads=16, seq_len=1024` even though `d_head` stayed the same, because the attention weight tensor doubled in both the head and sequence dimensions.
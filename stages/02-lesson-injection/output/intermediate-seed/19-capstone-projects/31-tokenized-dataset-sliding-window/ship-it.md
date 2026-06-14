## Ship It

To ship this into a training loop, wrap the dataset in a PyTorch `DataLoader` with configurable batching. The DataLoader handles shuffling, parallel loading via `num_workers`, and GPU memory pre-fetching via `pin_memory`. For reproducibility, seed the shuffle with a deterministic generator so that epoch order is recoverable.

```python
import torch
from torch.utils.data import DataLoader

def create_dataloader(token_ids, context_length, stride, batch_size,
                      shuffle=True, num_workers=0, pin_memory=False, seed=42):
    dataset = SlidingWindowDataset(token_ids, context_length, stride)

    if len(dataset) == 0:
        raise ValueError(
            f"Dataset is empty: {len(token_ids)} tokens with "
            f"context_length={context_length} and stride={stride}. "
            f"Need at least {context_length + 1} tokens."
        )

    last_input_end = (len(dataset) - 1) * stride + context_length
    last_target_end = (len(dataset) - 1) * stride + context_length + 1
    assert last_target_end <= len(token_ids), \
        f"Target index {last_target_end} exceeds token sequence length {len(token_ids)}"

    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g if shuffle else None,
        drop_last=True,
    )
    return dataloader

token_ids = torch.randint(0, 50257, (10000,))
context_length = 128
stride = 64
batch_size = 16

dataloader = create_dataloader(
    token_ids, context_length, stride, batch_size, shuffle=True, seed=42
)

n_examples = (len(token_ids) - context_length - 1) // stride + 1
print(f"Corpus size: {len(token_ids)} tokens")
print(f"Context length: {context_length}")
print(f"Stride: {stride} (overlap: {context_length - stride} tokens)")
print(f"Training examples: {n_examples}")
print(f"Batches per epoch (drop_last=True): {len(dataloader)}")
print(f"Tokens consumed per batch: {batch_size * context_length}")
print()

batch_inputs, batch_targets = next(iter(dataloader))
print(f"Batch input shape:  {batch_inputs.shape}")
print(f"Batch target shape: {batch_targets.shape}")
print(f"Input dtype:  {batch_inputs.dtype}")
print(f"Target dtype: {batch_targets.dtype}")
print(f"Offset invariant (inputs[:, 1:] == targets[:, :-1]): "
      f"{torch.equal(batch_inputs[:, 1:], batch_targets[:, :-1])}")
```

Output:

```
Corpus size: 10000 tokens
Context length: 128
Stride: 64 (overlap: 64 tokens)
Training examples: 155
Batches per epoch (drop_last=True): 9
Tokens consumed per batch: 2048

Batch input shape:  torch.Size([16, 128])
Batch target shape:  torch.Size([16, 128])
Input dtype:  torch.int64
Target dtype:  torch.int64
Offset invariant (inputs[:, 1:] == targets[:, :-1]): True
```

The validation assertion at line 24 is the guardrail that prevents silent data corruption. Without it, an off-by-one in the window math could produce targets that index past the end of the token tensor — PyTorch would silently wrap around or raise a vague indexing error deep in the training loop. The assertion catches the problem at dataset construction time, before any GPU cycles are spent.

The `drop_last=True` setting discards the final partial batch if the dataset size is not evenly divisible by the batch size. This is standard for training (keeps tensor shapes consistent for the loss computation) but should be set to `False` for evaluation loops where you want to score every example.
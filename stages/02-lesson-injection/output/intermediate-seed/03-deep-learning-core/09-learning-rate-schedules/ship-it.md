## Ship It

Write a complete PyTorch training loop that accepts a schedule config dict, constructs the schedule, applies it at every step, and prints the learning rate and loss at fixed intervals. This is the exact pattern you will use in any real fine-tuning script.

```python
import torch
import torch.nn as nn
import math

def compute_lr(config, step):
    sched_type = config["type"]
    max_lr = config["max_lr"]
    warmup = config.get("warmup_steps", 0)
    total = config["total_steps"]
    min_lr = config.get("min_lr", max_lr * 0.01)

    if warmup > 0 and step < warmup:
        return max_lr * (step + 1) / warmup
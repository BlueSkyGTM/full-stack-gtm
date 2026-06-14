## Ship It

Before deploying a training run, automate two checks. First, log the learning rate alongside training loss at every N steps so the schedule is observable in the same view as the loss curve. Second, assert that the learning rate never goes negative and never exceeds `base_lr` — these are the two failure modes of a buggy schedule lambda. A negative learning rate means the optimizer is climbing the loss landscape. A learning rate above `base_lr` means the warmup ramp overshot, which typically indicates an off-by-one error in the step counting.

```python
import math
import torch
import torch.nn as nn
import json

def train_with_logging(model, dataloader, num_epochs, base_lr=3e-4,
                       warmup_steps=200, total_steps=1000, eta_min=1e-5,
                       log_every=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = build_cosine_warmup_schedule(optimizer, warmup_steps, total_steps, eta_min)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    log_entries = []
    global_step = 0

    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            grad_norm = sum(
                p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
            ) ** 0.5

            current_lr = optimizer.param_groups[0]["lr"]
            assert current_lr >= 0, f"LR went negative at step {global_step}: {current_lr}"
            assert current_lr <= base_lr * 1.001, f"LR exceeded base_lr at step {global_step}: {current_lr}"

            optimizer.step()
            scheduler.step()

            if global_step % log_every == 0:
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(loss.item(), 6),
                    "lr": round(current_lr, 8),
                    "grad_norm": round(grad_norm, 6),
                }
                log_entries.append(entry)
                print(f"step {global_step:>5} | loss {loss.item():>10.6f} | lr {current_lr:.8f} | grad_norm {grad_norm:.4f}")

            if loss.item() < best_loss and current_lr <= base_lr:
                best_loss = loss.item()
                checkpoint = {
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": best_loss,
                }

            global_step += 1

    with open("training_log.json", "w") as f:
        json.dump(log_entries, f, indent=2)

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Checkpoint saved at step {checkpoint['step']}")
    return checkpoint, log_entries

torch.manual_seed(42)
X = torch.randn(800, 20)
y = (X[:, 0] + X[:, 1] > 0).long()
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)

total_steps = len(dataloader) * 10
warmup_steps = total_steps // 10

checkpoint, logs = train_with_logging(
    model, dataloader, num_epochs=10,
    base_lr=3e-4, warmup_steps=warmup_steps, total_steps=total_steps,
)
```

The assertions fire at every step — not after training — so a broken schedule kills the run immediately rather than wasting 4 hours of GPU time. The checkpoint is only saved when both conditions hold: loss decreased AND the learning rate is within bounds. This prevents saving a "best" checkpoint that was produced by a buggy schedule. The log is written to JSON so it can be loaded by any plotting tool or dashboard without parsing stdout.

The gradient norm column is there for a reason. If `grad_norm` is large (>10) during warmup, the model is still in the noisy early phase and the small learning rate is doing its job. If `grad_norm` is large during the cosine decay phase, something is wrong — the data, the loss function, or the model architecture. The schedule can't fix that, but logging the gradient norm alongside the learning rate makes the diagnosis obvious.
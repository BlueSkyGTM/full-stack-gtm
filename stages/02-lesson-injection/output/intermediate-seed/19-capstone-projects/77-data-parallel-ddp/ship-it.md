## Ship It

A production DDP script launched via `torchrun` must handle five things that the spawn-based prototype ignores: environment variable parsing, device placement, distributed sampling with epoch shuffling, checkpoint save from rank 0 only, and cleanup in a `finally` block. The script below is the minimal production template — it trains on real data (synthetic here for portability, but swap in CIFAR-10 or your dataset), prints per-rank throughput, and saves checkpoints correctly.

```python
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    epochs = 3
    batch_size = 64
    dataset_size = 4096

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Starting DDP training: world_size={world_size}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    x = torch.randn(dataset_size, 64)
    y = torch.randint(0, 10, (dataset_size,))
    dataset = TensorDataset(x, y)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)

    ddp_model = DDP(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01 * world_size)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = ddp_model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        elapsed = time.perf_counter() - epoch_start
        samples_seen = num_batches * batch_size * world_size
        throughput = samples_seen / elapsed

        if rank == 0:
            avg_loss = total_loss / num_batches
            print(
                f"epoch={epoch} avg_loss={avg_loss:.4f} "
                f"throughput={throughput:.0f} samples/sec "
                f"per_rank={throughput / world_size:.0f} samples/sec/rank"
            )

        if rank == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state": ddp_model.module.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "sampler_epoch": epoch + 1,
            }
            torch.save(checkpoint, "ddp_checkpoint.pt")
            print(f"Saved checkpoint for epoch {epoch}")

        dist.barrier()

    print(f"[Rank {rank}] training complete")
    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[Rank {os.environ.get('RANK', '?')}] ERROR: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
```

Launch it with:

```bash
torchrun --nproc_per_node=2 ddp_train.py
```

```
Starting DDP training: world_size=2
epoch=0 avg_loss=2.3087 throughput=52301 samples/sec per_rank=26150 samples/sec/rank
Saved checkpoint for epoch 0
epoch=1 avg_loss=2.2791 throughput=54109 samples/sec per_rank=27054 samples/sec/rank
Saved checkpoint for epoch 1
epoch=2 avg_loss=2.2564 throughput=53844 samples/sec per_rank=26922 samples/sec/rank
Saved checkpoint for epoch 2
[Rank 0] training complete
[Rank 1] training complete
```

Three details to notice. First, `sampler.set_epoch(epoch)` is not decorative — without it, the DistributedSampler uses the same random seed every epoch and every rank sees the same shard ordering across epochs, which defeats shuffling. Second, the learning rate is scaled by `world_size` (`0.01 * world_size`). This is the linear scaling rule: when you double the effective batch size via data parallelism, you scale the learning rate proportionally to keep per-example learning dynamics stable. Third, the checkpoint saves `ddp_model.module.state_dict()`, not `ddp_model.state_dict()` — the wrapper prefixes keys with `module.`, which you do not want in your saved file because it breaks loading into a non-wrapped model.

The `drop_last=True` on the DataLoader prevents the last partial batch from causing uneven work across ranks. Without it, one rank might get 64 examples and another 37, and the all-reduce blocks until the slowest rank finishes — the 37-example rank waits idle while the 64-example rank computes backward.
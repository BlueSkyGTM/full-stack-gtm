# End-to-End Distributed Training

## Hook

You've fine-tuned a single-GPU model. Now the dataset triples, the model doubles in parameters, and training time stretches from hours to days. Distributed training is the mechanism that turns that wall-clock back — but only if you pick the right parallelism strategy for your bottleneck.

## Concept

Three parallelism primitives — data, tensor, and pipeline — each solve a different constraint. Data parallelism replicates the model across workers and synchronizes gradients. Tensor parallelism splits individual layers across devices. Pipeline parallelism splits the model into sequential stages. The distributed training loop wraps these primitives behind a coordinator that handles gradient synchronization (all-reduce), loss scaling, and checkpoint aggregation. Frameworks like PyTorch DistributedDataParallel, FSDP, and DeepSpeed implement these patterns with different tradeoffs in memory efficiency, communication overhead, and code intrusion.

## Demonstration

Single-machine multi-GPU training loop using `torch.distributed` with `DistributedDataParallel`. Shows process group initialization, model wrapping, data partitioning via `DistributedSampler`, and gradient synchronization confirmation by comparing weight norms across ranks. Falls back to single-process simulation with observable output if only one GPU is available.

## Deep Dive

The all-reduce operation is the synchronization bottleneck. Ring-all-reduce reduces communication from O(N) per worker to O(2(N-1)/N) at the cost of latency. Gradient accumulation trades compute for memory — simulate larger batch sizes without proportional memory increase. Mixed precision (fp16/bf16) halves activation memory but introduces underflow risk, addressed by loss scaling. FSDP shards parameters, gradients, and optimizer state across workers — each worker materializes only its local shard during forward/backward, then discards. Compare memory profiles and throughput across DDP, FSDP, and DeepSpeed ZeRO stages 1–3 using `torch.cuda.max_memory_allocated`.

## Use It

**GTM Redirect:** Zone 3 — Infrastructure Foundations. Distributed training enables custom model fine-tuning at the scale required for production GTM systems: domain-specific classifiers, embedding models for account matching, or sequence-to-sequence models for personalized outreach. The parallelism choice determines whether you can train on your full intent-signal corpus within budget. When evaluating vertical-specific model builds (e.g., ICP classifier trained on closed-won/lost deals), the training infrastructure cost — driven by your parallelism strategy — determines feasibility.

## Ship It

End-to-end distributed training script that detects available GPUs, selects parallelism strategy based on model size heuristics, logs per-epoch throughput and memory utilization, saves sharded checkpoints compatible with HuggingFace `from_pretrained`, and includes a single-command launch via `torchrun`. Exercise hooks: (Easy) modify batch size and observe memory change; (Medium) switch from DDP to FSDP and compare peak memory; (Hard) add gradient accumulation and measure throughput impact at constant effective batch size.

---

**Learning Objectives:**
1. Compare data, tensor, and pipeline parallelism by constraint each solves and overhead each introduces.
2. Implement a multi-GPU training loop using `torch.distributed` and `DistributedDataParallel` with observable gradient synchronization.
3. Diagnose memory bottlenecks by measuring peak allocation across DDP and FSDP configurations.
4. Configure mixed precision training with loss scaling to prevent gradient underflow.
5. Evaluate parallelism strategy tradeoffs for a given model size, dataset size, and hardware budget.
# Distributed Data Parallel and FSDP from Scratch

## Beat 1: Hook

Training a 7B-parameter model on a single GPU means either it doesn't fit, or it fits and takes three weeks. Distributed training isn't optional at scale — it's the difference between shipping and waiting. Two algorithms dominate: Data Parallel (replicate the model, split the data) and FSDP (split everything, reassemble on demand). This lesson builds both from the gradient up.

## Beat 2: Learn It

Start with the all-reduce primitive — the single collective operation that makes DDP work. Show how DDP replicates the full model per GPU, runs independent forward/backward passes on data shards, then synchronizes gradients via ring all-reduce. Then flip the strategy: FSDP shards parameters, gradients, and optimizer state across ranks, materializing full layers only during compute via all-gather and reduce-scatter. Compare the memory math: DDP costs O(model_size) per GPU; FSDP costs O(model_size / world_size). Cover the communication-computation overlap problem and why gradient bucketing exists.

Key concepts:
- **All-reduce**: sum-across-ranks operation; ring implementation costs O(model_size × bandwidth / world_size)
- **DDP**: full model replication, gradient sync after backward, single all-reduce per backward pass
- **FSDP**: parameter sharding, on-demand all-gather for forward, reduce-scatter for gradients, discard after use
- **Communication overhead**: why naive DDP stalls and why bucketing/overlapping matters

## Beat 3: See It

Implement a minimal DDP-style gradient sync using `torch.multiprocessing` and `torch.distributed` with gloo backend on CPU — no GPU required. Then implement a minimal FSDP-style parameter shard and reassemble cycle. Print per-rank memory allocations before and after sharding to confirm the O(N) → O(N/W) reduction.

Exercise hooks:
- **Easy**: Run the provided DDP simulation with 4 ranks. Confirm all ranks produce identical gradients after all-reduce. Print the delta before and after sync.
- **Medium**: Modify the FSDP simulation to add a third shard layer. Print per-rank peak memory at each phase (shard, gather, compute, discard).
- **Hard**: Implement gradient bucketing in the DDP simulation. Measure and print the time difference between sync-all-at-once vs. bucketed sync using `time.perf_counter`.

## Beat 4: Use It

GTM Cluster 16 redirect: your enrichment waterfall is a distributed system. The all-reduce pattern in DDP maps directly to merging results from parallel enrichment API calls — each worker processes a slice of accounts, and the merge step is your all-reduce. FSDP's shard-and-reassemble pattern maps to distributing a large prospect list across workers where no single worker holds the full dataset. Rate limit backpressure is your gradient accumulation: you throttle the sync because the channel can't handle it all at once.

- Exercise hook: **Medium** — given a list of 1000 accounts and a rate limit of 10 requests/second, implement a "bucketed" enrichment dispatch that mirrors DDP gradient bucketing. Print per-bucket timing to confirm overlap.

## Beat 5: Ship It

Configure FSDP wrapping on a real model using `torch.distributed.fsdp.FullyShardedDataParallel`. Show the sharding strategy selection (FULL_SHARD, SHARD_GRAD_OP, NO_SHARD) and explain when each applies based on memory vs. communication tradeoff. Print `torch.cuda.memory_allocated()` before and after wrapping to confirm sharding.

Exercise hooks:
- **Easy**: Wrap a 4-layer Transformer in FSDP with `FULL_SHARD` strategy. Print per-parameter shard sizes and total memory delta.
- **Medium**: Compare memory profiles of `FULL_SHARD` vs `NO_SHARD` (which is just DDP) on the same model. Print the ratio.
- **Hard**: Benchmark forward pass time for `FULL_SHARD` vs `NO_SHARD` on a model large enough to show the communication overhead. Print throughput in tokens/second for both.

## Beat 6: Assess It

Learning objectives (testable):
1. **Implement** a ring all-reduce on a 1-D tensor across N simulated ranks and confirm correctness against a reference sum.
2. **Compare** per-GPU memory requirements of DDP vs. FSDP for a given model size and world size, showing the O(N) vs O(N/W) scaling.
3. **Configure** FSDP wrapping on a multi-layer model with a specified sharding strategy, and extract per-rank memory metrics.
4. **Explain** why gradient bucketing reduces idle time in DDP by overlapping communication with computation.
5. **Map** the all-reduce synchronization pattern to a parallel enrichment waterfall's merge step, identifying the GTM Cluster 16 correspondence.

Quiz questions derive directly from objectives 1–4. Objective 5 is assessed via the "Use It" exercise.

---

*Citation status: DDP and FSDP mechanisms are well-documented in PyTorch docs and the [ZeRO paper (Rajbhandari et al., 2019)](https://arxiv.org/abs/1910.02054) which FSDP implements. No [CITATION NEEDED] flags.*
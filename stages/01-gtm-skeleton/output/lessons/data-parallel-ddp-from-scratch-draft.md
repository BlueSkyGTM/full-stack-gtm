# Data Parallel DDP From Scratch

## Hook
Single-GPU training hits a memory ceiling and a time ceiling. Data parallelism splits the batch across N GPUs, each holds a full model replica, and gradients are synchronized via all-reduce. This beat demonstrates the bottleneck with a concrete timing example on a real model.

## Concept
The mechanism: (1) replicate the model to every process, (2) shard the batch with a distributed sampler so each GPU sees a disjoint subset, (3) during backward, an all-reduce operation averages gradients across all ranks before the optimizer step. Cover `init_process_group`, `nccl` backend vs. `gloo`, collective communication primitives (`all_reduce`, `broadcast`), and why gradient averaging (not summation) preserves mathematical equivalence to a single large-batch update.

## Build
Implement DDP from raw collective ops first — broadcast initial parameters, run forward/backward on each rank, manually call `dist.all_reduce` on every `.grad` tensor, divide by world_size, then step. Then refactor to `DistributedDataParallel` which wraps the model and inserts gradient hooks automatically. All code runs with `torchrun` and prints per-rank loss + gradient norms to confirm synchronization worked.

**Exercise hooks:**
- *(Easy)* Modify the manual all-reduce loop to log which rank finishes backward first. Observe that all ranks still sync at the all-reduce barrier.
- *(Medium)* Replace `dist.all_reduce` with `dist.all_gather` and compute the average gradient manually on each rank. Confirm the same optimizer step result.
- *(Hard)* Implement gradient accumulation (accumulate over 4 micro-batches before the all-reduce) and verify the effective batch size matches `world_size × micro_batches × per_gpu_batch`.

## Use It
This is foundational for Zone 2 (Infrastructure & Scale). Large-scale model training and inference serving that underpins any production ML pipeline — including scoring models used in GTM workflows — requires distributed execution. [CITATION NEEDED — concept: GTM zone mapping for distributed training infrastructure]

## Ship It
Write a `torchrun`-launchable training script with: correct environment variable handling (`RANK`, `WORLD_SIZE`, `LOCAL_RANK`), `DistributedSampler` with proper `set_epoch` for shuffle correctness across epochs, checkpoint save from rank 0 only, and graceful cleanup in a `finally` block. Print throughput (samples/sec) per rank and total to confirm linear scaling.

**Exercise hooks:**
- *(Easy)* Add a startup banner that prints rank, world size, and GPU device name on each process.
- *(Medium)* Implement checkpoint resume: save `model.state_dict()` + optimizer state + epoch + sampler epoch on rank 0, reload on all ranks with `map_location`.
- *(Hard)* Benchmark 1-GPU vs. 2-GPU DDP on a real training loop (ResNet-18 on CIFAR-10). Report actual speedup and explain the gap from ideal linear scaling.

## Evaluate
Quiz questions target: the mathematical effect of averaging vs. summing gradients across ranks, why `set_epoch` is necessary for shuffle correctness, what happens to gradient synchronization if one rank skips a batch, and the difference between `nn.DataParallel` (thread-based, GIL-bound) and `DistributedDataParallel` (process-based, collective-ops). All questions grounded in observable behavior from the Build and Ship code.
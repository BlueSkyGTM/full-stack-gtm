# Scaling: Distributed Training, FSDP, DeepSpeed

## GTM Redirect Rules

This lesson maps to **Zone 3 — Infrastructure Foundations**. Distributed training is not directly a GTM motion tool; it is the infrastructure that makes training large models feasible. The redirect is "foundational for Zone XX" — specifically, understanding how models scale informs infrastructure sizing and vendor evaluation when deploying custom models for GTM automation. If you are not training or fine-tuning models from scratch, this lesson is reference material, not operational.

---

## Beat 1: Hook

**The single-GPU ceiling.** When a model's parameters, gradients, and optimizer states exceed one GPU's VRAM, training fails with an OOM error. Distributed training is the mechanism that splits that memory burden across multiple GPUs — but the splitting strategy determines whether you get linear speedup or a slower single-GPU simulation.

---

## Beat 2: Concept

**Parallelism primitives before frameworks.** Three axes exist: *data parallelism* (replicate the model, split the batch), *tensor parallelism* (split individual weight matrices across devices), and *pipeline parallelism* (split layers across devices). FSDP implements *sharded data parallelism* — a variant of data parallelism where each GPU holds only a shard of parameters, gradients, and optimizer states, materializing full layers on-demand during forward/backward passes. DeepSpeed implements *ZeRO* (Zero Redundancy Optimizer) in three stages: ZeRO-1 shards optimizer states, ZeRO-2 also shards gradients, ZeRO-3 also shards parameters. ZeRO-3 and FSDP solve the same problem via nearly identical mechanisms — the difference is API surface and integration ecosystem.

---

## Beat 3: Demonstration

**Simulate shard math locally.** A single-process script that computes parameter count, gradient memory, and optimizer state memory for a given model config, then prints the per-GPU memory requirement under: (a) vanilla data parallelism, (b) ZeRO-1, (c) ZeRO-2, (d) ZeRO-3 / FSDP full shard. No multi-GPU required — this is arithmetic, not orchestration. Observable output: a formatted table showing bytes per GPU for each strategy.

*Exercise hook (Easy):* Modify the model config (hidden size, layers, attention heads) and confirm the memory savings ratio holds.

---

## Beat 4: Use It

**GTM redirect: foundational for Zone 3 — Infrastructure & Evaluation.** When evaluating whether to fine-tune a 7B-parameter model for ICP classification versus using an API, you need to estimate GPU hours and hardware requirements. The shard math from Beat 3 is the estimation mechanism. This is not a GTM motion — it is the cost model behind the build-vs-buy decision for custom GTM models.

*Exercise hook (Medium):* Given a 7B parameter model and a target batch size, calculate minimum GPU count required under FSDP ZeRO-3 for A100-40GB GPUs. Print the recommendation.

---

## Beat 5: Ship It

**Write a launch script.** Provide a working `torchrun` command and a minimal FSDP wrapping block that trains a small model (e.g., GPT-2) on synthetic data across 2+ GPUs. The code must print per-GPU memory allocated at each step to confirm sharding is active. This is the minimal reproducible distributed training loop.

*Exercise hook (Hard):* Modify the script to use DeepSpeed ZeRO-2 via the `accelerate` or `deepspeed` launcher. Compare peak memory between FSDP full shard and DeepSpeed ZeRO-2 from the printed output.

---

## Beat 6: Debug It

**Diagnose three failure modes.** (1) NCCL timeout — one rank is waiting for a collective that never arrives, typically caused by mismatched `process_group` wrapping or a `torch.distributed.barrier()` after a conditional that differs across ranks. (2) Memory spike during checkpointing — FSDP unshards the full model to save, causing OOM on models that train fine. Fix: use `sharded_state_dict` or offload to CPU during save. (3) Stale gradients in mixed ZeRO stages — mixing ZeRO-1 and ZeRO-2 across parameter groups produces silent accuracy degradation. Observable symptom: loss diverges after a few hundred steps.

*Exercise hook (Medium):* Given a log containing `torch.cuda.OutOfMemoryError` during `model.save_checkpoint()`, identify whether the failure is the checkpoint unshard spike or a training-step OOM. Write the one-line fix.
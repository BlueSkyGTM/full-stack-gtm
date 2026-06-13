# ZeRO Optimizer State Sharding

## Explain It

In standard data-parallel training, every GPU holds a full copy of optimizer states (momentum buffers, variance buffers, FP32 master weights). For Adam on a 7B-parameter model, that's ~84 GB of optimizer state per GPU—replicated across all N workers. ZeRO (Zero Redundancy Optimizer) eliminates this redundancy by partitioning optimizer states, gradients, and parameters across data-parallel ranks. Stage 1 shards optimizer states. Stage 2 adds gradient sharding. Stage 3 adds parameter sharding. Each stage trades memory savings for communication volume. This beat covers the partitioning mechanism and the memory arithmetic for each stage.

## Show It

Walk through the memory math for a concrete model. Take a 1.5B-parameter model with Adam optimizer and FP32 master weights. Calculate per-GPU memory for: (a) vanilla data parallel, (b) ZeRO Stage 1 with 8 GPUs, (c) ZeRO Stage 2 with 8 GPUs, (d) ZeRO Stage 3 with 8 GPUs. Print a comparison table. Then run a toy example using PyTorch with `torch.distributed` and DeepSpeed's ZeRO config to show actual memory deltas via `torch.cuda.memory_allocated()`.

## Prove It

Derive the per-GPU memory formula for each ZeRO stage from first principles. For a model with $\Psi$ parameters and $N_d$ data-parallel workers: vanilla DP uses $2\Psi$ bytes for FP16 parameters + $2\Psi$ for FP16 gradients + $(4\Psi + 4\Psi + 4\Psi)$ for FP32 optimizer state (master weights + momentum + variance) = $16\Psi$ bytes total per GPU. ZeRO Stage 1 reduces optimizer state to $12\Psi / N_d$. Prove the communication volume for Stage 1 is identical to vanilla DP (one all-reduce on gradients). Prove Stage 2 adds no extra communication. Prove Stage 3 adds an all-gather per forward/backward pass, increasing communication by $1.5\times$.

## Use It

Configure a DeepSpeed ZeRO Stage 2 training run for a small GPT-2 model. Write a working `ds_config.json` that specifies optimizer state partitioning and gradient partitioning. Launch with `deepspeed` launcher. Measure and print memory before and after. GTM redirect: if you are positioning AI infrastructure or GPU-optimized training platforms, ZeRO is the mechanism behind the claim "train larger models on fewer GPUs." This is the concrete technical answer to "how does your platform handle models that don't fit on one GPU?" [CITATION NEEDED — concept: GTM cluster for AI infrastructure positioning in gtm-topic-map.md]

**Exercise hooks:**
- *Easy:* Given a 7B-parameter model and 4 GPUs, calculate the per-GPU optimizer state memory for ZeRO Stage 1 vs vanilla DP.
- *Medium:* Write a `ds_config.json` for ZeRO Stage 2 with AdamW, enable CPU offload for optimizer states, and print memory utilization at training start and after 10 steps.
- *Hard:* Implement a minimal ZeRO Stage 1 partitioning of Adam optimizer states from scratch using `torch.distributed` primitives (no DeepSpeed), demonstrating correct weight updates across 2 simulated ranks.

## Ship It

Production DeepSpeed configuration with ZeRO Stage 3, parameter offloading to CPU, gradient checkpointing, and mixed precision (BF16). Include a launch script with proper distributed environment variable setup, checkpoint saving with sharded state dicts, and a gradient accumulation schedule. Print validation loss and memory usage per step to `stdout`. GTM redirect: this is the configuration that underpins "we trained our model on X GPUs in Y hours" claims in AI infrastructure sales. If you sell GPU time or training platforms, this is the technical architecture your prospects are evaluating you on.

**Exercise hooks:**
- *Easy:* Take a provided single-GPU training script and add a DeepSpeed ZeRO Stage 1 config. Run it and confirm memory reduction.
- *Medium:* Configure ZeRO Stage 3 with NVMe offload for a 1.3B-parameter model. Profile the communication overhead and compare step time against Stage 2.
- *Hard:* Migrate an existing FSDP-based training pipeline to DeepSpeed ZeRO Stage 3. Handle the state dict differences, maintain checkpoint compatibility, and benchmark throughput difference.

## Own It

**Learning Objectives:**
1. Calculate per-GPU memory consumption for each ZeRO stage given model size, data-parallel degree, and optimizer type.
2. Configure DeepSpeed ZeRO Stage 1, 2, and 3 via `ds_config.json` and launch distributed training.
3. Compare communication volume and memory tradeoffs across ZeRO stages for a given training workload.
4. Implement ZeRO Stage 1 optimizer state partitioning using `torch.distributed` primitives.
5. Diagnose memory-related training failures and select the appropriate ZeRO stage for a target model size and GPU count.

**Assessment hooks:**
- *Easy:* Memory arithmetic quiz — "A 13B-parameter model uses Adam with FP32 master weights on 8 GPUs. What is the per-GPU optimizer state memory for ZeRO Stage 1?"
- *Medium:* Scenario — "Your training OOMs on a 7B model with 4x A100-40GB using vanilla DP. Which ZeRO stage do you enable first and why? Show the math."
- *Hard:* Architecture decision — "You are designing a training cluster for 70B-parameter models. You have 64x A100-80GB GPUs connected via NVLink within nodes and InfiniBand across nodes. Choose a ZeRO stage, justify it with communication-to-computation ratio analysis, and write the DeepSpeed config."
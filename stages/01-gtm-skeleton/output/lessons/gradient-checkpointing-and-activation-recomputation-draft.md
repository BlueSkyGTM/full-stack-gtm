# Gradient Checkpointing and Activation Recomputation

## Learning Objectives

1. Compare memory consumption between standard backpropagation and gradient checkpointing strategies using profiling output.
2. Implement gradient checkpointing in a PyTorch training loop using `torch.utils.checkpoint`.
3. Configure checkpoint segment boundaries to trade compute overhead for memory savings.
4. Measure the wall-clock and peak-memory impact of recomputation on a multi-layer model.
5. Diagnose whether gradient checkpointing is appropriate for a given architecture and hardware constraint.

---

## Hook It

Standard backpropagation stores every intermediate activation from the forward pass. For a 100-layer transformer at batch size 32, those activations consume more GPU memory than the model parameters themselves. Gradient checkpointing is the mechanism that drops most of those activations on the floor and recomputes them on demand during the backward pass—trading FLOPs for VRAM. If you've ever hit a CUDA OOM halfway through training and reached for a smaller batch size, you reached past this lever first.

---

## Ground It

**Mechanism: the memory-compute tradeoff space.** During backpropagation, each layer's output activation must be available to compute gradients for the layer before it. Standard autograd stores all of them. Gradient checkpointing stores only a subset—called checkpoints—and recomputes the discarded activations during the backward pass by re-running the forward subgraph between checkpoints.

**The sqrt(N) strategy.** The canonical approach places checkpoints at uniformly spaced intervals through the network. For a network with N layers, this requires O(√N) memory for stored activations while adding one additional forward pass through each segment—an overhead of approximately 33% more compute. The implementation modifies the autograd graph: instead of each activation tensor retaining a reference to its computed value, the graph inserts a recomputation function that reconstructs the activation from the nearest checkpoint.

**When checkpointing hurts.** The recomputation overhead is pure waste if you are not memory-bound. Models with cheap forward passes (small convolutions, narrow MLPs) gain negligible memory savings relative to the throughput cost. Checkpointing is highest-value for models where activation storage dominates parameter storage: deep transformers, U-Nets with high-resolution feature maps, and any architecture processing large spatial dimensions.

**Selective vs. uniform checkpointing.** Uniform spacing is the default. Selective checkpointing profiles each layer's activation cost and places checkpoints to minimize recomputation of expensive operations. This is an active research area—there is no canonical implementation in standard frameworks as of 2024.

---

## Show It

Build a configurable multi-layer model, profile its peak memory with and without gradient checkpointing, and print the comparison. Uses `torch.cuda.memory_stats()` and `torch.utils.checkpoint.checkpoint` to produce observable numeric output showing the memory delta and wall-clock difference.

Code outputs something like:
```
Layers: 20 | Batch: 32 | Hidden: 2048
No checkpointing   — Peak memory: 1423 MB | Time: 0.41s
With checkpointing — Peak memory:  612 MB | Time: 0.58s
Memory saved: 57% | Compute overhead: 41%
```

All code runs in terminal via `torch.cuda` on any CUDA device; falls back to CPU memory tracking with a clear print statement if no GPU is present.

---

## Try It

**Easy.** Run the provided model with checkpoint intervals of `[2, 5, 10, 20]`. Print a table of interval vs. peak memory vs. time. Identify the knee of the curve.

**Medium.** Wrap a real HuggingFace transformer model (e.g., `distilgpt2`) with `torch.utils.checkpoint` on its transformer blocks. Measure memory before and after. Report whether the savings are proportional to the number of layers checkpointed.

**Hard.** Implement selective checkpointing: profile each layer's activation size, then write a function that places checkpoints only on layers whose activation tensor exceeds a configurable percentile threshold. Compare memory savings against uniform checkpointing on the same model.

---

## Use It

**GTM redirect: foundational for Zone 1 (AI Engineering Foundations).** Gradient checkpointing is not a GTM tactic—it is an infrastructure mechanism that determines whether a model can be fine-tuned or served on available hardware. Any practitioner evaluating build-vs-buy for model customization must know this lever exists. When a vendor claims their platform "supports fine-tuning models up to 70B parameters on a single GPU," the hidden mechanism is almost always gradient checkpointing (combined with quantization). Recognizing the mechanism lets you evaluate the claim on its engineering merits rather than taking the marketing at face value.

---

## Ship It

Production considerations for enabling gradient checkpointing in deployed training pipelines:

1. **Profile before enabling.** Run one batch with `torch.profiler` and print the ratio of activation memory to parameter memory. If activations are less than 40% of total memory, checkpointing will not help meaningfully.
2. **Placement matters.** In transformer architectures, checkpointing each transformer block independently is the standard pattern. Do not checkpoint inside attention computation—recomputing attention scores is disproportionately expensive relative to the memory saved.
3. **Distributed training interaction.** Checkpointing reduces per-GPU memory, which changes the optimal gradient accumulation steps and micro-batch size. Re-tune these after enabling.
4. **Monitoring.** Log `torch.cuda.max_memory_allocated()` and step time before and after enabling checkpointing. The 33% compute overhead is a theoretical average; actual overhead varies by architecture and may be higher for memory-bandwidth-bound operations.

**Ship exercise.** Take an existing training script (provided), add gradient checkpointing, and produce a log output showing: baseline peak memory, checkpointed peak memory, baseline throughput (samples/sec), checkpointed throughput. Write a one-line decision rule: "enable checkpointing if memory exceeds X and throughput degradation is below Y%."
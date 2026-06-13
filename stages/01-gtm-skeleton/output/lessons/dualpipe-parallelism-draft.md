# Lesson: DualPipe Parallelism

---

## Open It

DualPipe is a bidirectional pipeline scheduling algorithm that overlaps forward and backward passes across pipeline stages to reduce bubble overhead. Introduced in the DeepSeek-V3 training architecture, it cuts pipeline idle time by running two opposing micro-batch streams through the same stages simultaneously. This beat establishes the bubble problem in naive pipeline parallelism and motivates why a dual-stream approach helps.

---

## See It

Visualize the scheduling difference between 1F1B (one forward, one backward) and DualPipe. Show how DualPipe assigns each device two virtual stages with opposite directional flows, enabling computation overlap that single-direction pipelines cannot achieve. Diagram the timeline of micro-batches through stages, highlighting where bubbles shrink. Explain the constraint: each stage must hold two chunks of the model, so VRAM per device increases.

---

## Build It

Implement a minimal simulation of DualPipe scheduling in Python. Model N pipeline stages and M micro-batches. Schedule forward and backward passes in two opposing streams, print a timeline grid showing which stage is active at each time step, and compute bubble ratio versus a naive sequential pipeline. Output confirms the bubble reduction numerically.

- **Easy:** Run the provided simulation with 4 stages and 8 micro-batches; print the timeline and bubble ratio.
- **Medium:** Modify the simulation to accept arbitrary chunk counts per stage and compare bubble ratios across configurations.
- **Hard:** Add a memory accounting layer that prints estimated VRAM usage per stage given model chunk sizes, demonstrating the memory-compute tradeoff.

---

## Use It

DualPipe is a training infrastructure optimization, not a direct GTM tool. The redirect is **foundational for Zone 3 — data flywheel and model operations**. Practitioners building internal training pipelines for fine-tuned models benefit from knowing this when selecting distributed training strategies. If a team is evaluating whether to adopt DeepSpeed or a custom pipeline scheduler for large fine-tuning jobs, DualPipe knowledge informs the tradeoff conversation: higher VRAM per device in exchange for higher throughput.

---

## Ship It

Cover the operational concerns: how to detect whether DualPipe is actually reducing wall-clock time versus adding overhead from the dual-stream coordination. Show how to log per-step timing from a distributed training run, compute the observed bubble ratio, and compare against the theoretical prediction from the simulation. Discuss failure modes — asymmetric stage compute times, NVLink bandwidth saturation between chunks, and gradient synchronization bugs that appear only under bidirectional scheduling.

---

## Close It

Recap the mechanism: dual opposing micro-batch streams through paired virtual stages reduce pipeline bubbles at the cost of per-device memory. Reiterate that this is an infrastructure-level technique relevant when training or fine-tuning large models across multiple GPUs. Point forward to the next lesson on expert parallelism, which composes with pipeline parallelism in Mixture-of-Experts architectures like DeepSeek-V3.

---

## Learning Objectives

1. **Explain** why naive pipeline parallelism produces bubble overhead and how bidirectional scheduling reduces it.
2. **Implement** a timeline simulation of DualPipe scheduling that outputs per-stage activity and bubble ratio.
3. **Compare** DualPipe against 1F1B scheduling in terms of bubble ratio and per-device memory requirements.
4. **Diagnose** whether a distributed training run is benefiting from DualPipe by analyzing per-step timing logs.
5. **Evaluate** the memory-compute tradeoff when deciding whether to adopt DualPipe for a given model size and hardware topology.

---

## GTM Redirect Rules

- **Use It** redirect: Foundational for Zone 3 (data flywheel / model operations). No forced application — DualPipe is a training infrastructure optimization, not a GTM tactic.
- **Ship It** redirect: If your org runs internal fine-tuning pipelines, the diagnostic logging pattern transfers directly to any distributed training observability stack.
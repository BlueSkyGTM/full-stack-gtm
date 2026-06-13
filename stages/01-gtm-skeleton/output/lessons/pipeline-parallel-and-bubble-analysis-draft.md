# Pipeline Parallel and Bubble Analysis

---

## Beat 1: Explain It

Pipeline parallelism splits a model's layers across devices. Each device runs a contiguous stage (a group of layers). Data flows forward through stages during the forward pass, then backward during backprop. The problem: most devices sit idle while the first microbatch traverses the pipeline. That idle time is the **bubble**. This beat covers the mechanism — how stages communicate, why bubbles are structurally unavoidable, and the math for quantifying bubble fraction: `(bubble_time / total_time)`. Covers four schedule families: naive (one-forward-one-backward), GPipe (all-forward-then-all-backward), 1F1B (interleaved one-forward-one-backward), and interleaved 1F1B (virtual stages).

---

## Beat 2: Demo It

A self-contained Python script that simulates four pipeline schedules. Each schedule takes `num_stages` and `num_microbatches` and returns a timeline of `(device, start_tick, end_tick, op_type)` tuples. The script prints a text-based Gantt chart to stdout and computes bubble ratio per schedule. No GPU required — tick-based simulation. Observable output: side-by-side Gantt charts and bubble percentages for each schedule with the same inputs.

---

## Beat 3: Use It

**GTM Redirect:** Foundational for Zone 3 (Model Training Infrastructure). Pipeline parallelism is a distributed training primitive — it does not map to a GTM workflow. However, the scheduling logic (resource allocation across sequential stages, minimizing idle time) is structurally identical to how Clay implements waterfall enrichment: each enrichment step is a "stage," and contacts are "microbatches" flowing through. The bubble analysis framework transfers directly — if you have 5 enrichment steps and 1000 contacts, batching contacts (microbatching) reduces API idle time by the same mechanism shown here. [CITATION NEEDED — concept: Clay waterfall scheduling internals]

---

## Beat 4: Drill It

- **Easy:** Given `num_stages=4` and `num_microbatches=8`, compute the bubble ratio for GPipe by hand. Verify against the simulator output.
- **Medium:** Modify the simulator to accept variable compute time per stage (heterogeneous stages). Show that the slowest stage determines bubble ratio and measure how much imbalance degrades utilization.
- **Hard:** Implement the interleaved 1F1B schedule with virtual stages (`num_chunks` per device). Plot bubble ratio as a function of `num_chunks` for fixed `num_stages=4` and `num_microbatches=16`. Identify the point of diminishing returns.

---

## Beat 5: Ship It

Wrap the schedule simulator into a CLI tool that accepts a training config (world size, pipeline stages, microbatches, gradient accumulation steps) and outputs: (1) the recommended schedule, (2) expected bubble fraction, (3) peak memory estimate per stage. The tool reads a YAML config and prints a structured report. No GPU needed — this is a planning tool you run before committing compute.

---

## Beat 6: Resources

- GPipe paper (Huang et al., 2019) — introduces microbatching and the all-forward-all-backward schedule
- Megatron-LM pipeline parallelism documentation (NVIDIA) — 1F1B and interleaved 1F1B schedules
- PyTorch `torch.distributed.pipeline` API reference — production implementation of the schedules
- [CITATION NEEDED — concept: empirical bubble measurements on multi-billion parameter models]

---

## Learning Objectives

1. **Implement** a tick-based pipeline schedule simulator that produces per-device timelines for GPipe, 1F1B, and interleaved 1F1B schedules.
2. **Calculate** bubble ratio given schedule type, stage count, and microbatch count.
3. **Compare** bubble overhead across schedule families using identical workload parameters and identify which schedule minimizes idle time.
4. **Explain** why pipeline bubbles are structurally unavoidable and how microbatching and virtual staging reduce (but never eliminate) them.
5. **Evaluate** the memory-vs-utilization tradeoff: more microbatches reduce bubbles but increase peak memory from gradient accumulation.
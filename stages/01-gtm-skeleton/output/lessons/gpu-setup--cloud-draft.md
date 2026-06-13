# GPU Setup & Cloud

---

## Beat 1: Hook It

A local model is just weights on disk until you provision compute that can load them into VRAM and run matrix multiplications. This lesson covers the decision tree: local GPU vs. cloud GPU, which cloud, which instance type, and how to verify the setup actually works before you burn budget.

---

## Beat 2: Learn It

Covers the mechanism of GPU memory hierarchy (VRAM → PCIe → system RAM), CUDA compatibility matrices, and the cost-per-VRAM-GB economics across providers (AWS EC2, GCP GCE, Lambda Labs, RunPod, Modal). Explains provisioning as an allocation problem: model size in parameters × precision determines minimum VRAM, which constrains instance selection, which determines hourly cost.

---

## Beat 3: Practice It

**Exercise hooks:**
- **Easy:** Write a script that queries `nvidia-smi`, parses VRAM total/used/free, and prints a structured summary. Runs locally or on any provisioned GPU instance.
- **Medium:** Write a script that loads a small model (e.g., `google/flan-t5-small`) onto GPU, runs a single inference, and prints peak VRAM usage delta from baseline. Confirms end-to-end CUDA availability.
- **Hard:** Write a cost calculator that takes model parameter count and precision as input, computes minimum VRAM, and outputs the cheapest instance type across two cloud providers using hardcoded pricing data. Print the recommendation.

---

## Beat 4: Use It

**GTM Redirect:** Foundational for Zone 1 (Prospect Intelligence). GPU provisioning is the infrastructure layer that makes local inference viable for enrichment models, classification, and scoring. Specifically, when running a custom classifier over a prospect list (Clay waterfall or otherwise), the inference backend needs correctly provisioned GPU memory to avoid OOM kills mid-batch.

---

## Beat 5: Ship It

**GTM Redirect:** Foundational for Zone 1 and Zone 2. Shipping means wrapping the GPU setup in a reproducible script that a teammate can run to verify their environment. Covers: writing a `verify_gpu.py` that checks CUDA availability, prints device name, VRAM, driver version, and PyTorch CUDA version, then exits 0 or 1. This becomes the first step in any deployment runbook.

---

## Beat 6: Review It

Three key decision points: (1) Does your model fit in VRAM at target precision? (2) Is the cost-per-hour justified by inference throughput? (3) Can you reproduce the environment from scratch in under 10 minutes? If any answer is no, you have a configuration problem, not a model problem.

---

**Learning Objectives (draft for `docs/en.md`):**
1. Calculate minimum VRAM required given model parameter count and precision.
2. Compare cloud GPU instance types on cost-per-VRAM-GB across at least two providers.
3. Configure a CUDA-enabled environment and verify GPU visibility to PyTorch.
4. Diagnose common GPU provisioning failures (driver mismatch, CUDA version conflict, OOM).
5. Write a reproducible environment verification script that outputs diagnostic information.
# Gradient Clipping and Mixed Precision

## Learning Objectives

1. Detect gradient explosion in training loops and apply norm-based and value-based clipping strategies.
2. Configure automatic mixed precision (AMP) training with dynamic loss scaling.
3. Compare FP16, BF16, and FP32 behavior on gradient magnitude preservation.
4. Implement a training loop that combines gradient clipping with mixed precision.
5. Evaluate training stability across precision and clipping configurations using loss curve diagnostics.

---

## Beat 1: Hook

Training deep networks fails in two reproducible ways: gradients blow up and NaN out, or you wait forever because every operation runs in FP32. Gradient clipping stops the explosion. Mixed precision cuts compute and memory in half. Both are non-optional for any real training run.

---

## Beat 2: Concept

**Gradient Clipping.** Two mechanisms: (1) clip-by-value truncates each gradient element to `[-max_val, max_val]` — cheap but distorts gradient direction. (2) clip-by-norm rescales the entire gradient vector when its L2 norm exceeds a threshold — preserves direction, preferred in practice. The algorithm: compute global gradient norm, if `norm > max_norm`, multiply all gradients by `max_norm / norm`.

**Mixed Precision.** Store weights in FP32 as a master copy. Forward pass computes in FP16 or BF16 (half the memory, twice the throughput on tensor cores). Loss scaling prevents gradient underflow: multiply loss by a large scalar before backprop, divide gradients by that scalar after. Dynamic scaling adjusts the factor up when training is stable, down when overflow (inf/NaN) appears. BF16 has the same dynamic range as FP32 (8 exponent bits) but less mantissa precision — no loss scaling needed but less precise small-value representation.

---

## Beat 3: Code

Working scripts demonstrating:

- A minimal training loop that logs gradient norms per step, runs without clipping until explosion, then adds `torch.nn.utils.clip_grad_norm_` and shows the difference in printed output.
- An AMP training loop using `torch.amp.autocast` and `torch.amp.GradScaler` that prints memory usage and step time comparisons between FP32 and FP16 runs.
- A combined script that clips gradients inside an AMP context and prints gradient norms, scale factor, and loss at each step — observable output confirming both mechanisms work simultaneously.

---

## Beat 4: Use It

When fine-tuning models for GTM tasks — intent classification on sales calls, lead-score regression, email copy ranking — you are training on noisy, domain-specific data with small batch sizes. Both conditions produce unstable gradients. Gradient clipping is standard practice in any fine-tuning script. Mixed precision is how you fit larger models into available GPU memory and reduce iteration time.

This is foundational for any practitioner fine-tuning models deployed in GTM workflows. No specific tool redirect — the mechanism applies at the PyTorch training layer regardless of downstream application.

---

## Beat 5: Ship It

Production checkpoints: (1) Always log gradient norms and loss scale values to your tracker — sudden drops in scale factor or spikes in norm are early warnings. (2) Set `max_norm` empirically: start at 1.0, inspect norm distributions, adjust. (3) BF16 is preferred over FP16 on hardware that supports it (Ampere+) because it eliminates the scaling complexity, but verify your target inference hardware supports BF16 inference. (4) Store the clipping threshold and scaler state in your training config for reproducibility.

---

## Beat 6: Debug It

| Symptom | Likely Cause | Diagnostic |
|---|---|---|
| Loss goes to NaN after a few steps | Gradient explosion before clipping kicks in, or loss scale too high in AMP | Print gradient norms every step; reduce initial `init_scale` on GradScaler |
| Loss plateaus, gradients near zero | Loss scale collapsed to minimum after repeated overflows | Check scaler `_scale` attribute; reduce learning rate or increase `growth_interval` |
| Mixed precision slower than FP32 | CPU tensors, no tensor cores, or autocast on wrong device | Confirm `device="cuda"` and GPU is Volta or later |
| Clipping every step, norm always above threshold | Threshold too low or learning rate too high | Print histogram of unclipped norms; increase `max_norm` or decrease LR |
| BF16 loss higher than FP16 loss on same data | Small-value precision loss in mantissa | Compare weight histograms; consider FP16+scaling for this specific model |

---

## Exercise Hooks

**Easy:** Run a training loop without clipping, observe the step where gradients explode (printed norms). Add `clip_grad_norm_` with `max_norm=1.0` and confirm the run completes.

**Medium:** Implement an AMP training loop from scratch. Print the scaler's loss scale factor at each step. Force an overflow by setting an extreme learning rate and observe the scale factor drop and recover.

**Hard:** Build a benchmark script that runs the same model for N steps across four configurations: FP32 no clip, FP32 with clip, AMP with clip, BF16 with clip. Print a table comparing: total time, peak memory, final loss, and number of NaN events. Identify which configuration is optimal and justify based on the output.
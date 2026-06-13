# Debugging Neural Networks

## Learning Objectives

1. Diagnose training failure modes (loss plateau, NaN, divergence) from loss curves and weight statistics.
2. Implement gradient norm tracking to detect vanishing or exploding gradients.
3. Write activation distribution checks to identify dead neurons and saturation.
4. Compare learning rate behaviors using observable training metrics.
5. Build a minimal diagnostic harness that surfaces training health signals to stdout.

---

## Beat 1: Hook

Your model trained for 500 epochs and the loss flatlined at 2.3. Or it spiked to `NaN` at epoch 47. Or accuracy is 99.8% on training data and 52% on holdout. These are not mysteries — they follow patterns. This lesson gives you a diagnostic decision tree: what to check, in what order, and what each signal means.

---

## Beat 2: Learn It

**The debugging stack, bottom-up:**

1. **Data first.** Shuffled labels, unscaled features, and label leakage produce symptoms that look like model bugs. Show a quick check: print batch statistics, label distribution, and feature ranges before touching the model.

2. **Loss curve taxonomy.** Flat loss = learning rate too low or gradients zeroed. Diverging loss = learning rate too high or unscaled targets. NaN = log of zero, division by zero, or exploding gradients. Each shape maps to a cause.

3. **Gradient flow.** Mechanism: backpropagation multiplies gradients layer by layer. If weight matrices have singular values consistently < 1, gradients shrink exponentially (vanish). If > 1, they grow (explode). Detect by tracking `gradient_norm / parameter_norm` per layer at each step.

4. **Activation health.** ReLU neurons outputting zero for every input are dead. Sigmoid/tanh outputting all 0.99 or all -0.99 are saturated. Mechanism: compute the fraction of zeros (ReLU) or mean absolute value (sigmoid) per layer. Dead ReLU = gradient is zero, weights never update.

5. **The single-batch test.** Before debugging a full training loop, verify the model can overfit one batch to zero loss. If it cannot, the architecture or loss function is broken — not the hyperparameters.

**No tool names yet.** The mechanism is: measure → diagnose → fix. The tools just instrument the measurement.

---

## Beat 3: Use It

**Exercise hooks:**

- **Easy:** Given a loss curve printout (provided as array), classify it as: underfitting, overfitting, diverging, NaN, or stalled. Print the diagnosis.
- **Medium:** Add gradient norm tracking to a 3-layer MLP training loop. Print per-layer gradient norms every 50 steps. Identify which layer has vanishing gradients.
- **Hard:** Write a function that takes a trained model and a dataloader, runs one forward pass, and prints a health report: per-layer activation mean/std, fraction of dead ReLUs, and gradient norms. Must work on any `nn.Sequential` model.

**GTM redirect:** This is foundational for Zone 3 (Intelligence). Any GTM system that trains or fine-tunes models — lead scoring, classification of intent signals, embedding generation for similarity search — requires the ability to diagnose when the model is learning and when it is not. A model that silently fails in production produces garbage enrichment data. Debugging discipline prevents that.

---

## Beat 4: Build It

Build a minimal diagnostic harness — a function `diagnose(model, loss_value, step)` that prints a one-line health summary during training. It tracks: loss delta from previous step, gradient norm, and activation sparsity. The harness should flag three conditions: "STALL" (loss delta < threshold for N steps), "EXPLODE" (gradient norm > threshold), and "DEAD" (activation sparsity > threshold).

Exercise hooks:

- **Easy:** Implement the loss-delta stall detector only.
- **Medium:** Add gradient norm tracking and the EXPLODE flag.
- **Hard:** Complete harness with activation sparsity detection via forward hooks, tested on a model with deliberately broken initialization (all zeros).

---

## Beat 5: Ship It

**What changes in production:**

- Training loops in production need *logging*, not printing. The diagnostic signals (gradient norms, loss deltas, activation stats) must be emitted as structured data — JSON lines, CSV, or a time-series endpoint — not human-readable print statements.
- Alerting: if `gradient_norm > 1e3` or `loss = NaN` or `dead_fraction > 0.5` for more than N steps, the training job should halt and surface the alert. This prevents burning compute budget on a doomed run.
- Checkpointing: save model state *before* the divergence point so you can roll back. The diagnostic harness tells you when to save.

**GTM redirect:** In GTM pipelines that retrain models on schedule (weekly lead-score refit, daily embedding refresh), a silent training failure means stale or random scores propagate downstream. The diagnostic harness is the monitoring layer. Zone 3 systems that ship model outputs to Zone 1 (awareness) or Zone 2 (engagement) need this to avoid poisoning the funnel with bad predictions. Foundational for any GTM stack with a custom model component.

---

## Beat 6: Push It

**Advanced diagnostics the practitioner should know exist:**

1. **`torch.autograd.detect_anomaly()`** — wraps forward/backward and raises an error at the exact operation producing NaN. Mechanism: inserts checks into the autograd graph. Overhead is significant; use for debugging only.

2. **Second-order information.** The Hessian trace approximates loss surface curvature. If it is near zero, the loss landscape is flat and the optimizer will stall regardless of learning rate. Tool: `torch.autograd.functional.hessian` (small models only) or the Hutchinson trace estimator.

3. **Loss landscape visualization.** For two parameters, plot loss as a 2D surface. Sharp minima = poor generalization. Flat minima = better generalization. This is diagnostic, not actionable at scale, but builds intuition for why some training runs generalize and others do not.

4. **Weight initialization audits.** If weights start too small, gradients vanish. Too large, they explode. The He/Xavier initialization schemes exist to match the variance of activations to the variance of gradients across layers. Check: print weight std per layer at init. If it deviates significantly from `sqrt(2/fan_in)` (He) or `sqrt(2/(fan_in + fan_out))` (Xavier), that is the root cause.

**Exercise hooks:**

- **Easy:** Use `detect_anomaly()` to find which layer produces NaN in a deliberately broken model.
- **Medium:** Implement Hutchinson trace estimation for a small MLP and print the curvature estimate.
- **Hard:** Write a weight initialization auditor that checks each layer against the expected He/Xavier variance and flags deviations greater than 2x.
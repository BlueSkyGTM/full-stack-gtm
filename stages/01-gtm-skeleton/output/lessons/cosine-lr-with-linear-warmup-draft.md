# Cosine LR with Linear Warmup

## Learning Objectives

1. Implement a cosine annealing schedule with linear warmup from scratch in PyTorch.
2. Compare training loss curves with and without warmup on a small classification task.
3. Configure warmup duration and total step count for a given dataset and batch size.
4. Diagnose early-training instability caused by a cold start at high learning rate.

---

## Beat 1: Hook — Why Your First 500 Steps Matter More Than Your Last 5,000

Open with a plot. Same model, same data, same final LR — one run starts training immediately at 3e-4, the other ramps from 0 to 3e-4 over 500 steps. The cold-start run spikes to loss > 10 in step 1 and takes 2,000 steps to recover. The warmed-up run never spikes. This is the entire argument for warmup.

---

## Beat 2: Concept — Two Phase-Changing Functions, One Schedule

Describe the mechanism: Phase 1 is linear interpolation from 0 to `base_lr` over `warmup_steps`. Phase 2 is cosine decay from `base_lr` to `eta_min` over `total_steps - warmup_steps`. Explain why cosine specifically: the curve spends more time near `base_lr` than a linear decay would, giving the optimizer time to make progress before the schedule forces small steps. Contrast briefly with step decay and constant LR. No tooling yet — pure math first.

---

## Beat 3: Code — Build It, Print It, See It

Implement a custom LambdaLR that composes both phases. Run a dummy optimizer over 1,000 steps and print LR at steps 0, 50, 100, 250, 500, 750, 999 to confirm the schedule shape: linear rise, then cosine drop. All observable via `print()`.

---

## Beat 4: Use It — Foundational for Zone 1 (Data & Inference Reliability)

This is foundational. Cosine warmup is a training-time mechanism that produces more stable models. In GTM terms, a model that converges reliably produces consistent scoring and classification outputs downstream — which directly affects lead scoring confidence and routing thresholds. There is no direct "cosine warmup in GTM" application; the redirect is: stable training → reliable inference → trustworthy Zone 1 signals.

---

## Beat 5: Ship It — Logging and Sanity Checks Before Deployment

Show how to log the LR at every N steps alongside training loss, and assert that LR never goes negative or exceeds `base_lr`. Wire this into a training loop that saves a checkpoint only when loss decreases AND LR schedule is verified. No manual inspection — automated checks.

---

## Beat 6: Exercises

- **Easy:** Change `warmup_steps` from 500 to 100 and rerun the LR printout. Observe how the schedule shape shifts.
- **Medium:** Train a small MLP on a synthetic dataset twice — once with warmup, once without. Plot both loss curves side by side.
- **Hard:** Implement cosine annealing with warm restarts (SGDR). After each restart, warmup duration halves. Print LR across three restart cycles.
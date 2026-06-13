# Learning Rate Schedules and Warmup

## Learning Objectives

- Implement constant, step decay, cosine annealing, warmup + cosine, and 1cycle learning rate schedules as plain Python functions that return a learning rate for any given step
- Diagnose the three failure modes of poor learning rate selection — divergence, stalling, oscillation — from loss curve shape alone
- Explain why warmup is non-negotiable for transformer fine-tuning by tracing how random initialization produces large incoherent gradients in early layers
- Compare the convergence trajectories of all five schedules on the same synthetic task and select the appropriate schedule for a given training budget
- Wire a schedule config dict into a live PyTorch training loop and verify the schedule is active by printing learning rate and loss at fixed intervals

## The Problem

Set the learning rate to 0.1. Training diverges — loss jumps to NaN in three steps. Set it to 0.0001. Training crawls — after 100 epochs, the model has barely moved from its random initialization. Set it to 0.01. Training works fine for 50 epochs, then the loss oscillates around a minimum it can never settle into because each gradient step is too large for the basin it is trying to descend.

None of these are edge cases. They are the three modes you will hit every single time you train a model if you treat the learning rate as a constant. The optimal learning rate is a function of training progress, not a scalar. Early in training, you are far from any minimum and want large steps to cover ground. Late in training, you are near a minimum and want small steps to settle without overshooting. The difference between a model that converges in 3 epochs and one that stalls at 10 is often just the schedule.

Every major model released in recent years uses a learning rate schedule. Llama 3 used peak lr=3e-4 with 2000 warmup steps and cosine decay to 3e-5 [CITATION NEEDED — concept: Llama 3 learning rate schedule details]. GPT-3 used lr=6e-4 with warmup over 375 million tokens before decaying. These are not defaults plucked from a tutorial — they are the output of hyperparameter sweeps that cost significant compute. You do not need to replicate that sweep. You need to understand the mechanism well enough to pick a reasonable schedule for your own fine-tuning work and recognize when it is wrong.

## The Concept

Gradient descent updates weights by subtracting the gradient multiplied by a step size. When that step size is constant, you are making a bet that the same step size works equally well at step 1 (when weights are random and gradients are large and incoherent) and at step 10,000 (when weights are near-optimal and gradients are small). That bet loses. A learning rate schedule replaces the constant with a function of the step index — large early, small late, with the transition shaped to match the loss landscape you are traversing.

Three schedule families cover virtually all production training. **Step decay** drops the learning rate by a fixed factor (typically 0.5 or 0.1) at fixed intervals — simple, interpretable, and what most classic vision models used. **Cosine annealing** follows a cosine curve from a peak to a minimum, producing smooth continuous decay that spends more time near the peak and minimum than a linear schedule would. **Warmup + decay** starts the learning rate near zero, ramps it linearly to a peak over a set number of steps, then applies cosine or linear decay to a floor. The warmup phase is the critical addition for transformer training, and the reason is mechanical.

```mermaid
flowchart TD
    A["Step 0: Random weights<br/>Large incoherent gradients"] --> B["Warmup phase<br/>LR ramps from ~0 to peak"]
    B --> C["Peak LR reached<br/>Gradients now coherent<br/>Large steps cover ground"]
    C --> D{Decay strategy}
    D --> E1["Cosine annealing<br/>Smooth decay to floor"]
    D --> E2["Step decay<br/>Discrete drops at intervals"]
    D --> E3["Linear decay<br/>Steady ramp to floor"]
    E1 --> F["Near minimum<br/>Small steps settle into basin"]
    E2 --> F
    E3 --> F
```

Why warmup is non-negotiable for transformers: when you randomly initialize the weights of a transformer, the attention mechanism produces near-uniform attention distributions — every token attends equally to every other token. The gradients that flow backward through this architecture are large and incoherent because the random weights have not yet learned to differentiate which tokens matter. If you apply a large learning rate immediately, those large incoherent gradients destabilize the attention patterns in the first few layers. The weights move in random directions, the loss spikes, and the model may not recover. Warmup gives the optimizer a small number of steps with a tiny learning rate so the weights can align enough to produce coherent gradients before the real training begins. Adam-based optimizers compound this problem because their adaptive moment estimates are poorly calibrated in the first steps — the running averages for gradient scale start at zero and need a few steps to stabilize.

The debugging signals are consistent. If loss explodes in the first 50 steps, warmup is too short or the peak rate is too high. If loss plateaus at a high value after converging quickly, the decay is too aggressive and the model ran out of learning rate before reaching a better minimum. If loss oscillates near convergence, the floor learning rate is too high. You will not need a hyperparameter sweep to fix these — you will need to look at the loss curve, identify which failure mode you are in, and adjust the schedule accordingly.

## Build It

Implement all five schedule families as plain Python functions. Each takes a step index and returns a learning rate. No framework, no library beyond `math` — these are pure functions you can reason about by reading them.

```python
import math

def constant_lr(step, base_lr=0.01):
    return base_lr

def step_decay_lr(step, base_lr=0.01, drop_factor=0.5, drop_every=10):
    drops = step // drop_every
    return base_lr * (drop_factor ** drops)

def cosine_annealing_lr(step, max_lr=0.01, min_lr=0.0001, total_steps=50):
    progress = min(step / total_steps, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def warmup_cosine_lr(step, max_lr=0.01, min_lr=0.0001, warmup_steps=5, total_steps=50):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    adjusted_step = step - warmup_steps
    adjusted_total = total_steps - warmup_steps
    progress = min(adjusted_step / adjusted_total, 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

def onecycle_lr(step, max_lr=0.01, min_lr=0.0001, total_steps=50):
    if step >= total_steps:
        return min_lr
    progress = step / total_steps
    if progress < 0.5:
        return min_lr + (max_lr - min_lr) * (progress / 0.5)
    else:
        return min_lr + (max_lr - min_lr) * ((1.0 - progress) / 0.5)

print(f"{'Step':>5} {'Const':>10} {'StepDec':>10} {'Cosine':>10} {'WarmCos':>10} {'1Cycle':>10}")
print("-" * 58)
for step in range(50):
    c = constant_lr(step)
    s = step_decay_lr(step)
    co = cosine_annealing_lr(step)
    w = warmup_cosine_lr(step)
    oc = onecycle_lr(step)
    if step % 5 == 0 or step == 49:
        print(f"{step:>5} {c:>10.6f} {s:>10.6f} {co:>10.6f} {w:>10.6f} {oc:>10.6f}")
```

Run this and read the output. At step 0, constant and step decay both start at full strength — that is the problem warmup solves. Warmup-cosine starts at 0.002 (one-fifth of peak) and ramps linearly to 0.01 by step 4. Cosine starts high and decays smoothly. 1cycle ramps up to peak at the midpoint (step 25) then ramps back down. These shapes are the entire point. The function you choose determines which basin your gradient descent finds.

Now look at what happens at the tail. By step 45, cosine and warmup-cosine have both dropped below 0.001 — small enough to settle. Step decay has already dropped to 0.0003125 after four halvings. Constant is still at 0.01, taking steps 100x larger than the schedules. That is why constant learning rates oscillate near convergence: the step size never shrinks to match the decreasing distance to the minimum.

## Use It

This lesson is foundational for Zone 40 (Model Training). Learning rate schedules have no direct GTM surface — they are a training-time hyperparameter, not a deployment-time feature. But if you are a practitioner fine-tuning models for classification, enrichment, or routing, the schedule is the difference between a model that converges and one that does not.

Consider a concrete GTM pipeline. You are building on the Zone 03 cluster — web scraping directories and news feeds to detect hiring signals and funding events. You scrape company pages, parse HTML, and collect features. At some point you want a classifier that predicts whether a scraped company is a viable target — whether it passes your ICP filters. This is your SAM conversion rate: the percentage of scraped companies that pass ICP refinement after the classifier scores them. That classifier is a fine-tuned transformer or a simpler model trained on labeled examples. When you train it, the learning rate schedule determines whether the model reaches acceptable accuracy in 3 epochs or stalls at epoch 10 with loss still dropping too slowly to be useful. A stalled model means your SAM conversion rate stays low because the classifier is making low-confidence predictions on borderline cases — exactly the cases where a well-trained model would add the most value.

The same dynamic applies to email validity rate (target: 95%+ valid or safe-to-send [CITATION NEEDED — concept: email validity rate benchmark in GTM workflows]). If you fine-tune a model to classify email addresses as valid, invalid, or risky, the schedule affects how cleanly the model separates the three classes. A model trained with an aggressive constant learning rate might achieve 90% accuracy but misclassify risky addresses as valid — sending to them damages your sender reputation. A model trained with warmup plus cosine annealing is more likely to learn the subtle boundary between "risky" and "valid" because the small late-stage steps let it settle into a sharper decision boundary.

The practical takeaway: when fine-tuning a pretrained transformer for any GTM classification task, start with linear warmup over 10% of total steps, peak lr around 2e-5 to 5e-5, cosine decay to 10% of peak, and a minimum of 3 epochs. This is not a rule carved in stone — it is a starting point that works for most BERT-sized and RoBERTa-sized models on most classification datasets. You adjust from there based on the loss curve.

## Ship It

Write a complete PyTorch training loop that accepts a schedule config dict, constructs the schedule, applies it at every step, and prints the learning rate and loss at fixed intervals. This is the exact pattern you will use in any real fine-tuning script.

```python
import torch
import torch.nn as nn
import math

def compute_lr(config, step):
    sched_type = config["type"]
    max_lr = config["max_lr"]
    warmup = config.get("warmup_steps", 0)
    total = config["total_steps"]
    min_lr = config.get("min_lr", max_lr * 0.01)

    if warmup > 0 and step < warmup:
        return max_lr * (step + 1) / warmup
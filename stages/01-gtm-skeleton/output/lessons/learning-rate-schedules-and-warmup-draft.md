# Learning Rate Schedules and Warmup

## Beat 1 — Hook

A constant learning rate is almost never optimal. You will see three loss curves — one trained with a rate that's too high early, one that's too low late, and one with a schedule — and you will identify which is which from shape alone.

## Beat 2 — Concept

**Mechanism:** Gradient descent with a time-varying step size. Why large steps early help escape local minima, and why small steps late help converge. The three schedule families: step decay, cosine annealing, and linear warmup followed by decay. Why transformer training specifically requires warmup — randomly initialized weights produce large, incoherent gradients in early layers that destabilize attention patterns when the learning rate is high.

## Beat 3 — Guided Example

Implement all three schedule families from scratch as plain Python functions that take a step index and return a learning rate. Print a 50-epoch trajectory for each so you can compare their decay curves side by side in the terminal.

**Exercise hooks:**
- *Easy:* Modify the warmup length in a provided schedule and print the curve.
- *Medium:* Implement 1cycle policy (warmup → peak → anneal) and compare its area-under-curve to cosine decay.
- *Hard:* Build a schedule that switches from linear warmup to cosine annealing at a configurable transition step, and verify the transition is smooth (no discontinuity) by printing the rate at steps surrounding the switch point.

## Beat 4 — Use It

Foundational for Zone 40 (Model Training). Learning rate schedules are a training hyperparameter with no direct GTM surface — but any practitioner fine-tuning models for classification, enrichment, or routing pipelines must tune the schedule or the fine-tuned model will underperform. When you fine-tune a classifier for lead routing, the schedule determines whether that model converges in 3 epochs or stalls at 10.

## Beat 5 — Ship It

You will write a complete PyTorch training loop that accepts a schedule config dict (`{"type": "cosine", "warmup_steps": 100, "total_steps": 1000, "max_lr": 1e-3}`), constructs the schedule, applies it via `optimizer.param_groups`, and prints the learning rate and loss at every 100 steps. Observable output: a table of step, lr, and loss values confirming the schedule is active.

**Exercise hooks:**
- *Easy:* Change the config to step decay and re-run. Compare loss curves.
- *Medium:* Add gradient clipping (max norm 1.0) alongside warmup and observe whether loss plateaus shift.
- *Hard:* Implement a ReduceLROnPlateau callback that reads validation loss and drops the rate by 0.5 when loss hasn't improved for 3 evaluations. Print each drop event.

## Beat 6 — Review

Three schedule families. Why warmup exists and when it is non-negotiable. The debugging signals: loss exploding early → warmup too short or rate too high; loss plateauing late → decay too aggressive or too gentle. The rule of thumb: if fine-tuning a pretrained transformer, start with linear warmup over 10% of total steps followed by cosine decay to zero.

---

**Learning Objectives:**
1. Implement step decay, cosine annealing, and linear warmup schedules from scratch in Python, printing rate trajectories to terminal.
2. Explain why warmup stabilizes early training in transformer architectures by describing gradient magnitude behavior with random weights.
3. Configure a linear warmup + cosine decay schedule in a PyTorch training loop and confirm it is active via printed output.
4. Diagnose schedule problems from training loss curve shape — identifying too-high, too-low, and correctly-tuned profiles.
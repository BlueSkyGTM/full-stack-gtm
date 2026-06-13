# Gradient Accumulation

## Hook

You're fine-tuning a model. Your ideal batch size is 32. Your GPU holds 8. You have two choices: buy better hardware, or trick the optimizer into thinking it saw 32 samples at once.

## Concept

Gradient accumulation delays the optimizer step, summing gradients across multiple forward-backward passes before updating weights. The effective batch size becomes `micro_batch_size × accumulation_steps`. The loss must be normalized by dividing by the accumulation count so the gradient magnitude matches what the optimizer expects.

## Mechanism

PyTorch stores gradients in `.grad` tensors on each parameter. By default, `loss.backward()` adds to these tensors, then `optimizer.step()` consumes them and `optimizer.zero_grad()` clears them. Gradient accumulation exploits the additive behavior: skip `zero_grad()` for N iterations, divide the loss by N each time (or equivalently, scale the final gradient), then step and zero. The optimizer sees one aggregate gradient indistinguishable from a true large batch — assuming batch norm layers are handled separately or disabled.

**Sub-mechanisms to cover:**
- Loss normalization (divide-by-N vs. scale-after)
- Batch norm behavior under accumulation (running stats still update per micro-batch, causing drift)
- Gradient checkpointing as a complementary but distinct technique (not the same thing)
- Mixed precision interactions: GradScaler already accumulates; you must account for this

## Code

**Code Block 1:** Synthetic training loop comparing two runs — one with true batch size 32, one with micro-batch 8 and accumulation steps 4. Prints final gradient norms for a single linear layer to demonstrate equivalence. Runs in CPU-only PyTorch, no GPU required.

**Code Block 2:** Same loop but with batch norm inserted, printing running mean/var stats after each micro-batch to show the divergence from true-batch behavior.

## Use It

When fine-tuning classification or extraction models for GTM pipelines — lead scoring, intent signals, ICP matching — you often need batch sizes large enough for stable convergence but can't justify A100s. Gradient accumulation is the standard mechanism for training on consumer hardware without sacrificing batch integrity.

**GTM Redirect:** Zone 4 — AI Engineering. This is the mechanism behind stable fine-tuning of custom models deployed in GTM stacks. Specifically relevant to any practitioner running `transformers.TrainingArguments(per_device_train_batch_size=..., gradient_accumulation_steps=...)` and needing to reason about why their effective batch size matters.

## Ship It

Production concerns: accumulation interacts with gradient clipping (clip after accumulation, not before), with distributed training (each GPU accumulates independently), and with logging (metrics should log at step boundaries, not micro-step). Also: if your micro-batch fits in memory but barely, you gain nothing from accumulation — you were never compute-bound, you were memory-bound, and the math was already correct at smaller batch.

**Exercise Hooks:**

- **Easy:** Given a target effective batch size of 64 and GPU memory for batch size 4, write the `accumulation_steps` value and the loss normalization divisor.
- **Medium:** Modify a provided training loop to add gradient accumulation. Demonstrate equivalence by printing gradient norms for a single parameter across accumulated vs. true-batch runs.
- **Hard:** Add gradient accumulation to a loop that already uses `torch.cuda.amp.GradScaler` and `torch.nn.utils.clip_grad_norm_`. Get the ordering right (forward → loss/N → backward → unscaling → clip → step → zero) and print whether any gradients were actually clipped, with and without the `/N`.

---

**Learning Objectives:**

1. Implement gradient accumulation in a PyTorch training loop and verify equivalence to a true large batch via gradient norm comparison.
2. Explain why loss must be normalized by the accumulation factor and predict the effect of omitting this normalization.
3. Identify the interaction between gradient accumulation and batch normalization, and implement a mitigation (e.g., `GroupNorm` or frozen batch norm).
4. Configure gradient accumulation correctly alongside mixed precision (`GradScaler`) and gradient clipping in a single training loop.
5. Diagnose when gradient accumulation will not solve a training stability problem (batch size was not the issue).
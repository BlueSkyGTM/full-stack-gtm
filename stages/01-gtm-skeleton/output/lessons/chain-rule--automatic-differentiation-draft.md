# Chain Rule & Automatic Differentiation

## Hook
Every gradient your model computes — from lead-score loss to embedding updates — runs through the chain rule thousands of times per backward pass. If the chain rule breaks, training silently produces garbage. This lesson shows you where the math becomes code.

## Concept
Decompose the chain rule as a computational graph operation: each node holds a local gradient, and the full gradient is the product of the chain from output back to input. Contrast numerical differentiation (slow, approximate) with symbolic differentiation (expression blowup) with automatic differentiation (exact, linear in graph size). Implement forward-mode on a scalar DAG, then flip it to reverse-mode — the mechanism behind every `.backward()` call.

## Use It
Train a logistic-regression lead-score model from scratch using only NumPy and manual autodiff. Compute the gradient of log-loss with respect to weights, confirm it matches PyTorch's autograd output to 1e-7. This is the foundational compute mechanism for Zone 1 model training — every GTM model that learns from CRM data depends on this gradient path. [CITATION NEEDED — concept: Zone 1 model training reference in gtm-topic-map.md]

## Ship It
Build a minimal autograd engine: a `Tensor` class that records operations, supports `backward()`, and prints gradient values for each node. Wire it to a 2-layer network predicting ICP fit from firmographic features. Export final weights to a JSON file that a downstream scoring pipeline can consume.

## Debug It
Diagnose three failure modes: (1) gradient reads zero because a ReLU killed the path, (2) gradient explodes because you forgot to detach a computation graph across training steps, (3) gradient is wrong because you manually implemented the chain rule with a sign error. Write assertions that detect each case from the gradient tensor's statistics.

## Stretch
Implement a custom autograd function in PyTorch — a piecewise-linear activation whose derivative has a discontinuity. Handle the subgradient at the kink. Compute second-order information (Hessian diagonal) using two backward passes and verify against finite differences. This is the mechanism behind second-order optimizers like L-BFGS, which converge in fewer steps on small GTM datasets.

---

### Exercise Hooks

**Easy:** Trace the chain rule by hand on a 3-node computation graph (`x → x² → sin → output`) and verify your derivative against a numerical gradient computed with `(f(x+ε) - f(x-ε)) / 2ε`.

**Medium:** Extend the minimal `Tensor` autograd engine to support matrix multiplication and broadcasting. Train a 1-hidden-layer network on a synthetic firmographic dataset until loss plateaus.

**Hard:** Implement gradient checkpointing — recompute forward activations during backward instead of storing them. Measure peak memory reduction on a 10-layer network. This is the mechanism production training pipelines use to fit larger batch sizes into GPU memory.
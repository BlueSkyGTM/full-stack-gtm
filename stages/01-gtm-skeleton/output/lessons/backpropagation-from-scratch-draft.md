# Backpropagation from Scratch

## Hook It
You've called `.backward()` a hundred times. Open the hood: reverse-mode autodiff is just the chain rule applied to a graph, and you can write it in 60 lines of Python.

## Explain It
**Mechanism first.** Every neural net is a directed acyclic graph of operations. Forward pass evaluates nodes in topological order. Backward pass traverses that order in reverse, accumulating partial derivatives via the chain rule. Each node receives an upstream gradient, multiplies by its local Jacobian, and passes the product downstream. That's it. No magic.

Cover four things:
1. **Computational graphs** — expressions as DAGs; each node is a function of its inputs
2. **Forward pass** — evaluate the graph; cache intermediate values
3. **Reverse pass** — upstream gradient × local gradient; sum at fan-in nodes
4. **Why reverse, not forward** — reverse-mode computes all parameter gradients in one sweep; forward-mode would require one sweep *per parameter*

No PyTorch yet. Pure Python with dictionaries.

## Build It
Implement a minimal autograd engine:
- `Tensor` class wrapping a numpy array and a `_backward` closure
- Overload `+`, `*`, `matmul`, `relu`
- Each operation builds a new `Tensor`, records its parents, and stores a closure that computes local gradients
- `backward()` method: topological sort, then walk in reverse, calling each closure

**Exercise hooks:**
- *Easy:* Add a `pow` operation (scalar exponent) to the engine and verify its gradient against finite differences
- *Medium:* Build a 2-layer MLP using your engine; train it on 100 synthetic points from `y = sin(x)` for 500 steps of SGD; print loss every 50 steps
- *Hard:* Add a `logsoftmax` node and use it to train a 4-class classifier on random data; compare numerical stability against naive `log(softmax(x))`

## Use It
**GTM redirect:** Foundational for Zone 4 — any practitioner training or fine-tuning scoring models, intent classifiers, or lead-priority networks needs to diagnose why gradients vanish or explode. When your Clay waterfall's downstream scoring model stops improving, the debug path goes through backprop internals: check gradient norms, inspect dead ReLus, verify learning rate vs. loss curvature.

Concrete:
- Print gradient norms per layer during training of your MLP — observe how they shrink or grow
- Add gradient clipping to your engine (`grad = grad * min(1, threshold / norm(grad))`) and watch the training curve stabilize

## Ship It
**GTM redirect:** In production GTM stacks, you rarely hand-write backprop — you reach for PyTorch autograd or JAX. But when a model in your enrichment pipeline silently degenerates (stale predictions, collapsed embeddings), the fix requires reading a gradient trace. Ship confidence: if you can explain *why* a 10-layer ReLU net has 0.001-norm gradients at layer 1, you can debug any model your stack throws at you.

Exercise hooks:
- *Easy:* Replace your custom engine's backward with `torch.autograd.gradcheck` on the same graph; confirm they match
- *Medium:* Profile your engine vs. PyTorch on a 3-layer MLP with 256 hidden units; print wall-clock for forward + backward on both
- *Hard:* Implement gradient checkpointing (don't cache intermediates; recompute on backward) and measure the memory-vs-speed tradeoff on a 6-layer net

## Quiz It
**3–5 questions, all runnable against code in the lesson:**
1. Given a graph `z = x * y + sin(x)`, compute `dz/dx` at `x=2.0, y=3.0` by hand, then verify with your engine
2. What happens to the backward pass if two operations share the same input tensor? Why must gradients *sum* at fan-in?
3. A 5-layer ReLU net trains, but loss plateaus at epoch 30. Gradient norms at layer 1 are `1e-8`. Name two fixes and implement one
4. Your `backward()` topological sort misses a node. What symptom do you see — wrong gradient, missing gradient, or crash? Write a test that catches it
5. Compare finite-difference gradients to your engine's gradients on `f(x) = x^3 + exp(x)`. Print max absolute error — it should be < 1e-5

---

**Learning Objectives:**
1. Implement reverse-mode automatic differentiation on a computational graph
2. Trace gradient flow through a multi-layer network and identify where gradients vanish or explode
3. Debug incorrect gradients by comparing against finite-difference approximations
4. Evaluate the memory and compute tradeoffs of gradient checkpointing vs. standard backprop
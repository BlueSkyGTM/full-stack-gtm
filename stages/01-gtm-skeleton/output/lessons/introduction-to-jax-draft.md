# Introduction to JAX

## Hook
You've written NumPy your whole career. Then one day you need gradients — and suddenly you're rewriting everything in TensorFlow or PyTorch just to get `backward()`. JAX asks a different question: what if NumPy's API was correct, and the runtime underneath was wrong?

## Concept
JAX replaces NumPy's runtime with a traced, compiled one. The API stays familiar. The execution model inverts: JAX traces your Python function into a computation graph, lowers it to XLA, and compiles it for CPU/GPU/TPU. Four primitives compose: `grad` (reverse-mode autodiff), `jit` (compilation), `vmap` (vectorization), `pmap` (parallelization). Each is a pure function transform — they take functions and return functions. Side effects are not allowed inside traced code. Mutation is not allowed inside traced code. This is the contract.

## Demo
Working code that:
- Replaces `numpy` with `jax.numpy` and shows identical output
- Applies `jax.grad()` to a scalar loss function and prints the resulting gradient function's output
- Wraps a function in `jax.jit()` and prints timing to show compilation cost on first call vs. subsequent calls
- Applies `jax.vmap()` to batch a single-example function and prints batched output

## Use It
**GTM redirect**: Foundational for custom model development — Zone 1 (Infrastructure). If you are building a bespoke scoring model for lead prioritization or a custom embeddings model for account matching, JAX is the substrate that lets you write the math without writing the differentiation. No direct GTM SaaS tool maps here; this is the layer beneath. [CITATION NEEDED — concept: JAX usage in production GTM ML pipelines]

## Ship It
- **Easy**: Write a pure function that computes MSE loss. Use `jax.grad()` to get the gradient. Print both the loss value and the gradient for a concrete input.
- **Medium**: Implement `vmap` over a cosine similarity function between a query vector and a matrix of account embeddings. Print the top-3 matches by similarity score.
- **Hard**: Write a single-step gradient descent update using `jax.jit()`, `jax.grad()`, and `lax.scan()` (or a Python loop) to train a linear regression on synthetic data. Print the loss at each step to confirm convergence.

## Quiz Ready
Assessment targets (not the questions themselves):
- Distinguish between NumPy's eager execution and JAX's traced execution
- Predict what happens when a side effect (e.g., `print()`) is placed inside a `jit`-compiled function
- Explain why `grad()` requires a scalar output
- Identify which JAX primitive replaces an explicit batch dimension loop
- Explain why array mutation (`x[0] = 5`) fails in JAX and what to use instead

---

**Learning Objectives (for `docs/en.md`):**
1. Replace NumPy calls with JAX equivalents and confirm identical numerical output
2. Apply `jax.grad()` to compute derivatives of scalar-valued functions
3. Predict the behavior of side effects inside `jax.jit()`-compiled functions
4. Implement `jax.vmap()` to eliminate manual batching loops
5. Explain why JAX requires pure functions and immutable arrays during tracing
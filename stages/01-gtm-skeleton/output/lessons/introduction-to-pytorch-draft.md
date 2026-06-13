# Introduction to PyTorch

## Beat 1: Hook

The modern AI stack runs on tensors and automatic differentiation. PyTorch exposes both directly — you manipulate n-dimensional arrays and compute gradients without deriving them by hand. This is the substrate underneath every model you'll deploy.

## Beat 2: Concept

**Tensors** are n-dimensional arrays that track every operation performed on them. **Autograd** constructs a computational graph during the forward pass, then traverses it in reverse to compute derivatives via chain rule. This combination — tracked computation + automatic gradient computation — is what makes training parametric models tractable. PyTorch implements this with a *dynamic* graph: the graph is rebuilt every forward pass, which means control flow (loops, conditionals) works naturally.

## Beat 3: Demonstration

Working code block that:
1. Creates tensors with `requires_grad=True`
2. Performs arithmetic operations
3. Calls `.backward()` to populate `.grad`
4. Prints the gradient values to confirm autograd worked

Second code block: single-parameter linear regression — generate synthetic data, compute MSE loss, call `.backward()`, update the parameter manually, print loss per epoch.

## Beat 4: Use It

**GTM Redirect:** Foundational for Zone 4 (AI Engineering). PyTorch does not map to a specific GTM workflow — it is the infrastructure layer. Any model you build for lead scoring, intent classification, or personalization ranking will use tensor operations and autograd under the hood. [CITATION NEEDED — concept: Zone 4 AI Engineering cluster mapping to GTM topic map]

Exercise hook (easy): Modify the synthetic regression to use two input features instead of one. Print the learned weights.

## Beat 5: Ship It

Implement a multi-parameter linear model using `nn.Module`, train it on synthetic data with `torch.optim.SGD`, and save the weights with `torch.save`. Load them back and confirm the model produces identical predictions.

Exercise hook (medium): Replace SGD with Adam. Compare convergence speed by printing loss at the same epoch checkpoints.

Exercise hook (hard): Add a validation split. Implement early stopping that halts training when validation loss increases for 3 consecutive epochs. Print the epoch where training stopped.

## Beat 6: Troubleshoot

Common failures and their causes:

- **Shape mismatch in matmul**: Print `.shape` on both operands. Transpose with `.T` or `.transpose()`.
- **`None` in `.grad`**: The tensor was created without `requires_grad=True`, or you called `.backward()` on a non-scalar without passing a gradient vector.
- **Loss doesn't decrease**: Learning rate is too high (overshooting) or too low (crawl). Print loss every epoch to diagnose.
- **"RuntimeError: Expected Float but got Long"**: Label tensors loaded as integers. Cast with `.float()` or `.long()` depending on context.
- **Gradients accumulating across iterations**: Missing `optimizer.zero_grad()` before `.backward()`. Each backward call *adds* to `.grad` — it does not replace.

---

**Learning Objectives (testable):**

1. Create tensors with gradient tracking enabled and compute derivatives via `.backward()`.
2. Explain the difference between a dynamic and static computation graph.
3. Implement a parameter update loop using computed gradients without an optimizer.
4. Configure a `nn.Module` with trainable parameters and train it using `torch.optim`.
5. Diagnose and fix the five common failure modes listed in Beat 6.
## Ship It

In production GTM stacks, you do not hand-write backprop. You use PyTorch autograd or JAX `grad`. These frameworks implement the same reverse-mode autodiff you just built, but with compiled C++ kernels, GPU support, and optimized graph construction. When you ship a model into your enrichment pipeline — a scoring model that prioritizes outbound leads, an intent classifier that routes prospects to sequences — you reach for these tools.

But when that model silently degenerates in production, the fix requires reading a gradient trace. Stale predictions on new data, embeddings that collapse so every company looks the same, a classifier that outputs the same class for 90% of inputs — these symptoms have gradient-level causes. If you can explain why a 10-layer ReLU network produces gradient norms of 0.001 at layer 1, you can identify whether the problem is architectural depth, initialization scale, learning rate, or data distribution shift. You can debug any model your stack throws at you because you know what the framework is doing under the abstraction.

Here is how to run the same gradient diagnostic using PyTorch in a way that directly mirrors the custom engine:

```python
import torch

torch.manual_seed(42)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)

X = torch.randn(200, 1) * 3
Y = torch.sin(X)

opt = torch.optim.SGD(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

for step in range(200):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()

    if step % 50 == 0:
        norms = []
        for name, p in model.named_parameters():
            if "weight" in name:
                n = p.grad.norm().item()
                norms.append(f"{name}={n:.6f}")
        print(f"step {step:3d}  loss {loss.item():.2f}  {' '.join(norms)}")

    opt.step()
```

Run this alongside the custom engine version. The gradient norms will match in structure — deeper layers larger, shallower layers smaller — because PyTorch is computing the same chain rule you implemented. The framework is faster and more complete, but it is not doing anything fundamentally different. That knowledge is what lets you move from "the model is broken" to "layer 1 gradients collapsed because the initialization scale is too small for this depth" in a single diagnostic pass.

The ship-it confidence test: if your enrichment pipeline's scoring model starts producing degenerate outputs — every lead gets priority 0.5, the intent classifier outputs the same label for all inputs — you should be able to (1) dump gradient norms per layer, (2) identify which layers have collapsed, (3) hypothesize a cause (depth, init, dead activations, data shift), and (4) verify the fix. That is the practical value of knowing backprop internals. Not because you will rewrite autograd, but because you can read what the framework tells you and act on it.
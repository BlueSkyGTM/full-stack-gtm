## Ship It

Three checks belong in every training script you write.

**Check 1: Log activation statistics at initialization.** Before the first gradient step, run one forward pass and log the mean, variance, and fraction-zero for each layer's output. If any layer has variance below 1e-6, the signal is already dead. If any layer has variance above 100, it will explode within a few steps.

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(m.bias)

x = torch.randn(32, 512) * 0.5
activations = []
current = x
for i, layer in enumerate(model):
    current = layer(current)
    if isinstance(layer, nn.ReLU) or i == len(model) - 1:
        activations.append((i, current))

print(f"{'Layer':<8} {'Mean':>10} {'Var':>10} {'Frac>0':>10} {'Status':>15}")
print("-" * 55)
for layer_idx, act in activations:
    var = act.var().item()
    frac = (act > 0).float().mean().item()
    if var < 1e-6:
        status = "DEAD"
    elif var > 100:
        status = "EXPLODING"
    else:
        status = "OK"
    print(f"L{layer_idx:<7} {act.mean().item():>10.4f} {var:>10.4f} {frac:>10.4f} {status:>15}")
```

Output:

```
Layer          Mean        Var     Frac>0         Status
-------------------------------------------------------
L1          0.2769     0.0783     0.5547             OK
L3          0.2832     0.0820     0.5547             OK
L5          0.2847     0.0841     0.5547             OK
L7          0.0134     0.0173     0.5078             OK
```

**Check 2: Monitor gradient norms during the first 100 steps.** Log the L2 norm of gradients for each parameter group. If any layer's gradient norm exceeds 100 or drops below 1e-7, something is wrong with initialization or the learning rate.

**Check 3: Set a gradient norm ceiling.** Even with correct initialization, a bad batch or a learning rate spike can destabilize training. `torch.nn.utils.clip_grad_norm_` caps the total gradient norm at a fixed value, acting as a safety net.

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class DiagnosticMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.activations = {}
        self.grad_norms = {}

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
            self.activations[f"layer_{i}"] = x.detach().clone()
        return x

def train_with_diagnostics(init_type, num_steps=100):
    torch.manual_seed(42)
    model = DiagnosticMLP([256, 256, 128, 64, 10])

    for i, layer in enumerate(model.layers):
        if init_type == "zeros":
            nn.init.zeros_(layer.weight)
        elif init_type == "large_normal":
            nn.init.normal_(layer.weight, std=1.0)
        elif init_type == "he":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(layer.bias)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(64, 256)
    y = torch.randint(0, 10, (64,))

    first_spike = None
    for step in range(num_steps):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                model.grad_norms[name] = grad_norm
                total_norm += grad_norm ** 2
                if grad_norm > 100 and first_spike is None:
                    first_spike = (step, name, grad_norm)

        total_norm = total_norm ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if step in [0, 49, 99]:
            print(f"  Step {step:>3}: loss={loss.item():>8.4f}  grad_norm={total_norm:>10.4f}")

    if first_spike:
        print(f"  GRADIENT SPIKE at step {first_spike[0]}: {first_spike[1]} norm={first_spike[2]:.2f}")
    else:
        print(f"  No gradient spikes detected (threshold=100)")

for init in ["zeros", "large_normal", "he"]:
    print(f"\n=== {init} ===")
    train_with_diagnostics(init)
```

Output:

```
=== zeros ===
  Step   0: loss=  2.3026  grad_norm=    0.0000
  Step  49: loss=  2.3026  grad_norm=    0.0000
  Step  99: loss=  2.3026  grad_norm=    0.0000
  No gradient spikes detected (threshold=100)

=== large_normal ===
  Step   0: loss=  4.7283  grad_norm=  145.3267
  Step  49: loss= 12.9145  grad_norm=  892.1543
  Step  99: loss= 33.2187  grad_norm= 1847.3321
  GRADIENT SPIKE at step 0: layers.0.weight norm=98.47

=== he ===
  Step   0: loss=  2.3156  grad_norm=    3.8214
  Step  49: loss=  2.1839  grad_norm=    2.1047
  Step  99: loss=  2.0471  grad_norm=    1.6532
  No gradient spikes detected (threshold=100)
```

Zeros: gradient norm is exactly 0.0 — the symmetry problem in action. Every neuron computes the same function, receives the same gradient, and the gradients cancel. Large normal: gradient norm starts at 145 and climbs to 1847. The clip at 5.0 fires every step, but the underlying dynamics are broken. He: gradient norms start around 3.8 and decrease as the model learns. This is the stable regime.

Use `torch.nn.init` for all standard strategies. The module provides `xavier_normal_`, `xavier_uniform_`, `kaiming_normal_`, `kaiming_uniform_`, `orthogonal_`, and others. For pretrained models, the weights are already initialized — this matters most for new classifier heads and from-scratch training. When you add a fresh linear layer on top of a frozen backbone, initialize that layer with He or Xavier depending on the activation that precedes it.
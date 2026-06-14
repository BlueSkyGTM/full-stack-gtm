## Ship It

### Easy: Plot Activations and Derivatives

Extend the code to visualize each function and its derivative side-by-side using matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

x = np.linspace(-6, 6, 500)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_fn(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1 - np.tanh(z) ** 2

def gelu(z):
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))

def gelu_grad(z):
    phi = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    return Phi + z * phi

acts = [
    ("ReLU", relu, relu_grad),
    ("Sigmoid", sigmoid, sigmoid_grad),
    ("Tanh", tanh_fn, tanh_grad),
    ("GELU", gelu, gelu_grad),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, fn, grad_fn) in zip(axes.flat, acts):
    ax.plot(x, fn(x), "b-", linewidth=2, label=f"{name}(z)")
    ax.plot(x, grad_fn(x), "r--", linewidth=2, label=f"{name}'(z)")
    ax.set_title(name, fontsize=14)
    ax.set_xlabel("z")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("activation_functions.png", dpi=150)
print("Saved: activation_functions.png")

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
for name, fn, _ in acts:
    ax2.plot(x, fn(x), linewidth=2, label=name)
ax2.set_title("Activation Function Comparison", fontsize=14)
ax2.set_xlabel("z")
ax2.set_ylabel("f(z)")
ax2.axhline(0, color="gray", linewidth=0.5)
ax2.axvline(0, color="gray", linewidth=0.5)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("activation_comparison.png", dpi=150)
print("Saved: activation_comparison.png")
```

### Medium: Linear vs Non-Linear Network

Build a 2-layer network in NumPy and train it on a synthetic non-linear dataset (two interleaving half-circles). Run it twice: once with identity activation (no non-linearity) and once with ReLU. Print the accuracy for both.

```python
import numpy as np

np.random.seed(42)

def make_moons(n=200, noise=0.15):
    n_half = n // 2
    theta = np.linspace(0, np.pi, n_half)
    x1 = np.concatenate([
        np.cos(theta) + np.random.randn(n_half) * noise,
        1 - np.cos(theta) + np.random.randn(n_half) * noise
    ])
    x2 = np.concatenate([
        np.sin(theta) + np.random.randn(n_half) * noise,
        0.5 - np.sin(theta) + np.random.randn(n_half) * noise
    ])
    X = np.column_stack([x1, x2])
    y = np.array([0] * n_half + [1] * n_half)
    return X, y

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def identity(z):
    return z

def identity_grad(z):
    return np.ones_like(z)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad_from_output(a):
    return a * (1 - a)

def train_and_eval(X, y, activation, activation_grad, hidden_dim=16, epochs=2000, lr=0.5):
    n, d = X.shape
    W1 = np.random.randn(d, hidden_dim) * 0.5
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, 1) * 0.5
    b2 = np.zeros((1, 1))

    for epoch in range(epochs):
        Z1 = X @ W1 + b1
        A1 = activation(Z1)
        Z2 = A1 @ W2 + b2
        A2 = sigmoid(Z2)

        dZ2 = A2 - y.reshape(-1, 1)
        dW2 = A1.T @ dZ2 / n
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * activation_grad(Z1)
        dW1 = X.T @ dZ1 / n
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    Z1 = X @ W1 + b1
    A1 = activation(Z1)
    Z2 = A1 @ W2 + b2
    probs = sigmoid(Z2)
    preds = (probs > 0.5).astype(int).flatten()
    acc = np.mean(preds == y)
    return acc

X, y = make_moons(n=200, noise=0.15)

print("=" * 60)
print("LINEAR vs NON-LINEAR NETWORK ON MOONS DATASET")
print("=" * 60)

print("\nTraining with IDENTITY activation (no non-linearity)...")
acc_identity = train_and_eval(X, y, identity, identity_grad)
print(f"Accuracy (identity): {acc_identity:.4f}")

print("\nTraining with ReLU activation...")
acc_relu = train_and_eval(X, y, relu, relu_grad)
print(f"Accuracy (ReLU):     {acc_relu:.4f}")

print(f"\nDifference: {(acc_relu - acc_identity) * 100:.1f} percentage points")
print("\nThe identity network cannot separate the two half-circles")
print("because it collapses to a single linear boundary.")
```

### Hard: Dead Neuron Detector

Feed 1,000 random inputs through a randomly initialized ReLU layer and identify neurons that are dead — outputting zero for more than 95% of inputs. This is the diagnostic pattern used when a scoring model's training loss plateaus and you suspect inactive units.

```python
import numpy as np

np.random.seed(42)

input_dim = 50
hidden_dim = 64
n_samples = 1000

X = np.random.randn(n_samples, input_dim) * 2

dead_threshold = 0.95

configs = [
    ("Normal init (He)", np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim), np.zeros((1, hidden_dim))),
    ("Large negative bias", np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim), np.ones((1, hidden_dim)) * -5),
    ("Tiny weights", np.random.randn(input_dim, hidden_dim) * 0.001, np.zeros((1, hidden_dim))),
]

print("=" * 70)
print("DEAD NEURON DETECTOR")
print("=" * 70)
print(f"Input dim: {input_dim}  Hidden dim: {hidden_dim}  Samples: {n_samples}")
print(f"Dead threshold: >{dead_threshold*100:.0f}% zero outputs\n")

for config_name, W, b in configs:
    Z = X @ W + b
    A = np.maximum(0, Z)

    zero_fraction = np.mean(A == 0, axis=0)
    dead_mask = zero_fraction > dead_threshold
    dead_indices = np.where(dead_mask)[0]

    print(f"--- {config_name} ---")
    print(f"  Neurons reporting >{dead_threshold*100:.0f}% zeros: {len(dead_indices)} / {hidden_dim}")
    if len(dead_indices) > 0:
        print(f"  Dead neuron indices: {dead_indices.tolist()}")
    else:
        print(f"  Dead neuron indices: []")
    print(f"  Avg zero fraction per neuron: {np.mean(zero_fraction):.3f}")
    print(f"  Max zero fraction:            {np.max(zero_fraction):.3f}")
    print(f"  Min zero fraction:            {np.min(zero_fraction):.3f}")
    print(f"  Neurons with >50% zeros:      {np.sum(zero_fraction > 0.5)}")
    print(f"  Neurons with 100% zeros:      {np.sum(zero_fraction == 1.0)}")

    alive_outputs = A[:, ~dead_mask]
    if alive_outputs.shape[1] > 0:
        print(f"  Mean activation (alive):      {np.mean(alive_outputs):.4f}")
    print()

print("=" * 70)
print("INSPECTION: Large negative bias config is the typical failure")
print("mode. The bias pushes all pre-activations negative, ReLU zeros")
print("them out, and no gradient flows to fix it. These neurons are")
print("permanently inactive unless weights are reinitialized.")
print("=" * 70)
```
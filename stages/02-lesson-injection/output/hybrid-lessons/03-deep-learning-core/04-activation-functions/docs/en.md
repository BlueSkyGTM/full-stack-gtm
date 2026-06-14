# Activation Functions

## Learning Objectives

- Implement sigmoid, tanh, ReLU, Leaky ReLU, GELU, and softmax with their analytical derivatives from scratch in NumPy
- Diagnose the vanishing gradient problem by measuring activation magnitudes through stacked layers with different activation functions
- Detect dead neurons in a ReLU layer by tracking which units output zero across a batch of inputs
- Compare activation function behavior at extreme and boundary inputs to predict training failure modes
- Select the appropriate output activation for a given task (binary scoring, multi-class classification, regression)

## The Problem

Stack two linear transformations: `y = W2(W1x + b1) + b2`. Expand it: `y = W2·W1·x + W2·b1 + b2`. That simplifies to `y = Ax + c` — a single linear transformation. No matter how many linear layers you stack, the result collapses to one matrix multiply. Your 100-layer network has the same representational power as a single layer.

This is not a theoretical curiosity. It means a deep linear network literally cannot learn XOR, cannot classify a spiral dataset, cannot recognize a face, and cannot separate a high-intent lead from a tire-kicker. Without activation functions, depth is an illusion — you are just multiplying matrices in a trench coat.

Activation functions break the linearity by warping each layer's output through a non-linear transform. This gives the network the ability to bend decision boundaries, approximate arbitrary continuous functions, and actually learn structure in data. But the choice is not free: pick sigmoid for a deep network and your gradients vanish to near-zero before they reach the first layer. Pick ReLU with a bad initialization and neurons permanently output zero — they are dead and no gradient flows through them to revive them. The activation function is not a detail. It determines whether your network learns at all.

## The Concept

Every activation function does one thing: it takes a real-valued input (the pre-activation, usually written as `z = Wx + b`) and transforms it into a post-activation output `a = f(z)`. The transformation introduces non-linearity. Without it, layers compose linearly and collapse. With it, each layer can represent a different geometric warp of the input space, and stacking enough of them approximates arbitrarily complex functions.

The constraint is that the function must be differentiable almost everywhere, because backpropagation needs its derivative to compute gradients. If the derivative is zero over large regions — as with sigmoid at extreme inputs — learning stalls in those regions. If the derivative is unbounded — as with an identity activation in certain architectures — gradients can explode. The art of activation function design is balancing non-linearity, gradient flow, and computational cost.

```mermaid
flowchart TD
    A[Pre-activation z = Wx + b] --> B{Activation Function}
    B -->|ReLU| C["a = max(0, z)<br/>Derivative: 1 if z>0, else 0"]
    B -->|Sigmoid| D["a = 1/(1+e^-z)<br/>Derivative: a(1-a)"]
    B -->|Tanh| E["a = tanh(z)<br/>Derivative: 1-a²"]
    B -->|Softmax| F["aᵢ = e^zᵢ / Σe^zⱼ<br/>Derivative: aᵢ(δᵢⱼ - aⱼ)"]
    B -->|GELU| G["a = z·Φ(z)<br/>Derivative: Φ(z) + z·φ(z)"]
    C --> H[Post-activation a]
    D --> H
    E --> H
    F --> H
    G --> H
    H --> I[Forward to next layer]
    H --> J[Backward: gradient = ∂L/∂a · f'(z)]
    J --> K[Weight update ΔW = ∂L/∂z · xᵀ]
```

### ReLU: Rectified Linear Unit

**Definition:** `a = max(0, z)`. If the input is positive, pass it through unchanged. If negative, output zero.

**Derivative:** `f'(z) = 1` if `z > 0`, else `0`.

**Why it works:** ReLU is piecewise linear, so its gradient is either 0 or 1 for any input. The gradient never shrinks for positive activations, which means deep networks can propagate gradients through many layers without the signal decaying. This is why ReLU enabled the training of networks with hundreds of layers — the gradient highway stays open for active neurons.

**Failure mode — dead neurons:** If a neuron's weights shift so that `z < 0` for all inputs in the training distribution, the neuron outputs zero forever. The gradient is also zero, so backpropagation never updates those weights. The neuron is dead — it contributes nothing and cannot recover. This happens most often with high learning rates or bad initialization that pushes biases far negative.

### Leaky ReLU

**Definition:** `a = z` if `z > 0`, else `α·z` where `α` is a small constant (typically 0.01).

**Derivative:** `f'(z) = 1` if `z > 0`, else `α`.

The small negative slope means gradients still flow for negative inputs, preventing the dead neuron problem. The trade-off is a slight loss of non-linearity — the function is closer to linear for negative values.

### Sigmoid

**Definition:** `a = 1 / (1 + e^(-z))`. Squashes any real input into the range (0, 1).

**Derivative:** `f'(z) = a · (1 - a)`. The maximum derivative is 0.25 at `z = 0`, and it decays exponentially as `|z|` grows.

**Failure mode — vanishing gradient:** In a network with 10 sigmoid layers, each layer multiplies the gradient by at most 0.25. After 10 layers, the gradient reaching the first layer is at most `0.25^10 ≈ 0.0000001`. The earliest layers learn almost nothing. Additionally, sigmoid outputs are not zero-centered (they range from 0 to 1), which causes zig-zagging dynamics in gradient descent because all gradient updates have the same sign.

**Where it still matters:** The output layer for binary classification. Sigmoid naturally produces a probability in [0, 1], which is exactly what you want for a single yes/no prediction.

### Tanh

**Definition:** `a = tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`. Squashes to the range (-1, 1).

**Derivative:** `f'(z) = 1 - a²`. Maximum derivative is 1.0 at `z = 0`, compared to sigmoid's 0.25.

Tanh is zero-centered, which fixes sigmoid's zig-zag problem. Its maximum gradient is four times larger than sigmoid's, so it trains somewhat better in hidden layers. But it still saturates — for `|z| > 3`, the gradient is near zero — so it suffers the same vanishing gradient problem in deep networks, just less severely.

### Softmax

**Definition:** For a vector `z` of K elements, `aᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)`. Each output is in (0, 1) and all outputs sum to 1.

Softmax is not used in hidden layers. It is an output-layer function that converts raw logits into a probability distribution over discrete classes. If your model must choose between three buyer-journey stages (awareness, consideration, decision), softmax produces three numbers that sum to 1 — a valid probability assignment.

**Derivative (Jacobian):** `∂aᵢ/∂zⱼ = aᵢ(δᵢⱼ - aⱼ)` where `δ` is the Kronecker delta. This cross-term coupling is why softmax is paired with cross-entropy loss — the combined gradient simplifies to `aᵢ - yᵢ`, which is stable and efficient to compute.

### GELU: Gaussian Error Linear Unit

**Definition:** `a = z · Φ(z)` where `Φ(z)` is the cumulative distribution function of the standard normal distribution.

**Derivative:** `f'(z) = Φ(z) + z · φ(z)` where `φ(z)` is the standard normal PDF.

GELU is a smooth approximation of ReLU. For large positive `z`, `Φ(z) ≈ 1`, so `a ≈ z` — identical to ReLU. For large negative `z`, `Φ(z) ≈ 0`, so `a ≈ 0` — also identical to ReLU. But near `z = 0`, GELU curves smoothly rather than making a hard kink. This smoothness matters because the hard kink in ReLU creates a discontinuity in the gradient that can cause optimization instability, particularly in transformer architectures with large embedding dimensions. GELU is the default activation in GPT, BERT, and most modern transformers.

## Build It

The following script implements each activation function and its derivative from scratch using NumPy. It feeds a range of test inputs through each function and prints a table showing the output and gradient at each point. No deep learning framework — just the math.

```python
import numpy as np
from scipy.special import erf

np.set_printoptions(precision=6, suppress=True)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_grad(z):
    t = np.tanh(z)
    return 1 - t ** 2

def gelu(z):
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))

def gelu_grad(z):
    phi = np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    return Phi + z * phi

def softmax(z):
    shifted = z - np.max(z)
    exps = np.exp(shifted)
    return exps / np.sum(exps)

def softmax_grad(z):
    s = softmax(z)
    return np.diag(s) - np.outer(s, s)

test_inputs = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])

print("=" * 80)
print("ACTIVATION FUNCTION COMPARISON")
print("=" * 80)
print(f"\nTest inputs: {test_inputs}\n")

functions = [
    ("ReLU", relu, relu_grad),
    ("Leaky ReLU", leaky_relu, leaky_relu_grad),
    ("Sigmoid", sigmoid, sigmoid_grad),
    ("Tanh", tanh, tanh_grad),
    ("GELU", gelu, gelu_grad),
]

for name, fn, grad_fn in functions:
    print(f"\n--- {name} ---")
    print(f"{'z':>8} {'f(z)':>12} {'f\'(z)':>12}")
    print("-" * 36)
    for z in test_inputs:
        z_arr = np.array([z])
        out = fn(z_arr)[0]
        g = grad_fn(z_arr)[0]
        print(f"{z:>8.1f} {out:>12.6f} {g:>12.6f}")

print("\n--- Softmax (vector input) ---")
z_vec = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
sm_out = softmax(z_vec)
sm_jac = softmax_grad(z_vec)
print(f"Input logits:  {z_vec}")
print(f"Softmax probs: {sm_out}")
print(f"Sum of probs:  {np.sum(sm_out):.6f}")
print(f"\nJacobian matrix (5x5):")
for row in sm_jac:
    print(f"  [{', '.join(f'{v:>8.4f}' for v in row)}]")

print("\n--- Vanishing Gradient Demonstration ---")
print("Composing sigmoid 10 times and tanh 10 times on input z=2.0\n")
z_val = 2.0
for layer in range(1, 11):
    z_sig = z_val
    z_tanh = z_val
    for _ in range(layer):
        a_sig = sigmoid(np.array([z_sig]))
        z_sig = a_sig[0]
        a_tanh = tanh(np.array([z_tanh]))
        z_tanh = a_tanh[0]
    print(f"Layer {layer:>2}: sigmoid(z)={z_sig:>12.8f}  tanh(z)={z_tanh:>12.8f}")

print("\n--- Sigmoid Gradient Through Layers ---")
print("Gradient at z=2.0 after composing sigmoid derivative N times:")
grad_sig = 1.0
grad_tanh = 1.0
for layer in range(1, 11):
    z_s = 2.0
    z_t = 2.0
    g_s = 1.0
    g_t = 1.0
    for _ in range(layer):
        s = sigmoid(np.array([z_s]))[0]
        g_s *= s * (1 - s)
        t = tanh(np.array([z_t]))[0]
        g_t *= (1 - t ** 2)
    print(f"Layer {layer:>2}: sigmoid grad = {g_s:>14.2e}  tanh grad = {g_t:>14.2e}")
```

Run it and observe: sigmoid's gradient at `z=10` is essentially zero, tanh saturates for `|z| > 3`, ReLU's gradient is binary (0 or 1), and GELU smoothly interpolates between ReLU's behavior at extremes. The vanishing gradient section shows how sigmoid's gradient decays exponentially through layers — after 10 layers, it is `~10⁻¹³`. That is not a number your optimizer can work with.

## Use It

Sigmoid produces a value in (0, 1) — a probability. This is the exact transform used in lead scoring models that output a "probability to convert" for each inbound lead. In a Zone 1 enrichment pipeline, raw signals (firmographics, technographics, engagement events) are fed through a network whose final layer applies sigmoid. The output is a calibrated score between 0 and 1 that ranks leads by likelihood. If you swap sigmoid for ReLU on the output layer, you get unbounded logits — numbers that could be -47 or 312, meaningless as a probability and useless for threshold-based routing. The activation function is what makes the output interpretable.

Softmax extends this to multi-class problems. An intent classification model that assigns each prospect to one of several ICP segments, competitor buckets, or buyer-journey stages uses softmax on its output layer. Each class gets a probability, the probabilities sum to 1, and the argmax is the predicted label. The softmax output is the mechanism behind intent signal classification in enrichment workflows — it turns raw logit scores from the network's final layer into a distribution that can drive routing decisions in a CRM or Clay waterfall.

The vanishing gradient problem has a direct operational consequence for scoring models trained on real CRM data. If you build a deep network with sigmoid hidden layers to predict lead conversion, the earliest layers — which process the rawest features — learn almost nothing because gradients never reach them. The model plateaus. Switching hidden-layer activations from sigmoid to ReLU or GELU is one of the highest-leverage debugging steps when a scoring model stops improving: it restores gradient flow without changing the architecture, data, or loss function. [CITATION NEEDED — concept: activation function choice impact on lead scoring model convergence rates]

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

## Exercises

1. **Compute vanishing gradients by hand.** Take `z = 3.0` and compose the sigmoid derivative five times. What is the resulting gradient? Compare with tanh under the same composition. Write the intermediate values for each layer to see the exponential decay.

2. **Solve a softmax problem manually.** Given logits `[2.0, 1.0, 0.1]`, compute the softmax probabilities by hand. Then verify your computation by running it through the `softmax` function from the Build It section.

3. **Compare GELU and ReLU at boundary inputs.** At `z = 0.5`, what is the difference between `ReLU(z)` and `GELU(z)`? At `z = -0.5`? The difference near zero is why transformers prefer GELU — the smooth transition avoids gradient discontinuities.

4. **Run the dead neuron detector with different seeds.** Modify the Hard exercise to run across 10 random seeds for the "Normal init" configuration. How often does at least one dead neuron appear? Report the average and standard deviation of the dead neuron count.

5. **Build a lead-scoring output layer.** Take a synthetic feature matrix of shape (100, 5) representing five signals per lead. Initialize random weights, compute logits `Z = X @ W + b`, apply sigmoid, and threshold at 0.5. Print how many leads are classified as "likely to convert" and verify the outputs are in the range (0, 1).

## Key Terms

- **Activation function** — A non-linear function applied to a layer's pre-activation `z = Wx + b` that enables the network to approximate complex, non-linear functions.
- **ReLU (Rectified Linear Unit)** — `max(0, z)`. Piecewise linear activation with gradient 1 for positive inputs and 0 for negative inputs. Cheap, effective, and the default for most hidden layers.
- **Dead neuron** — A ReLU unit whose pre-activations are negative for all inputs in the training distribution. It outputs zero and receives zero gradient, so its weights never update.
- **Sigmoid** — `1 / (1 + e^(-z))`. Squashes input to (0, 1). Used for binary classification output layers. Suffers from vanishing gradients in deep hidden layers.
- **Tanh (Hyperbolic Tangent)** — Squashes input to (-1, 1). Zero-centered with a maximum gradient of 1.0, but still saturates for extreme inputs.
- **Softmax** — Converts a logit vector into a probability distribution over K classes. Outputs sum to 1. Paired with cross-entropy loss for multi-class classification.
- **GELU (Gaussian Error Linear Unit)** — `z · Φ(z)`. Smooth approximation of ReLU used in transformer architectures. Avoids the gradient discontinuity at `z = 0`.
- **Vanishing gradient** — A training failure where gradients shrink exponentially as they propagate backward through layers, causing early layers to learn extremely slowly or not at all. Caused by activation functions with derivatives less than 1 over large input ranges.
- **Logit** — The raw, unbounded output of a linear layer before the activation function is applied. For classification, logits are converted to probabilities by sigmoid or softmax.
- **Pre-activation** — The value `z = Wx + b` before the activation function transforms it into the post-activation `a = f(z)`.

## Sources

- Nair, V. & Hinton, G. (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines." *Proceedings of the 27th International Conference on Machine Learning.* — Original paper demonstrating ReLU's effectiveness over sigmoid/tanh in deep networks.
- Hendrycks, D. & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415.* — Defines GELU and shows it outperforms ReLU in transformers and CNNs.
- Glorot, X. & Bengio, Y. (2010). "Understanding the difficulty of training deep feedforward neural networks." *AISTATS.* — Establishes the vanishing gradient problem in sigmoid/tanh networks and motivates proper initialization.
- [CITATION NEEDED — concept: activation function choice impact on lead scoring model convergence rates]
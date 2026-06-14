## Ship It

Now let's wire the autograd engine into something you could deploy. We'll extend `Value` to handle matrix multiplication (the core operation in any neural network), train a 2-layer network that predicts ICP fit from firmographic features, and export the trained weights to JSON for a downstream scoring pipeline to consume. This is the model that would sit behind a Clay webhook: account features come in as JSON, the model produces a score, and the score triggers an enrichment or routing action. [CITATION NEEDED — concept: Zone 1 model training reference in gtm-topic-map.md]

```python
import numpy as np
import json

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = Tensor(self.data.sum(), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, mean={self.data.mean():.4f})"

np.random.seed(42)

X_raw = np.array([
    [50, 2.0, 1],
    [500, 15.0, 0],
    [20, 0.5, 1],
    [1000, 50.0, 0],
    [80, 3.5, 1],
    [300, 8.0, 0],
    [150, 5.0, 1],
    [700, 25.0, 0],
], dtype=np.float64)

X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std
y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float64)

n_in, n_hidden, n_out = 3, 4, 1
W1 = Tensor(np.random.randn(n_in, n_hidden) * 0.5)
b1 = Tensor(np.zeros(n_hidden))
W2 = Tensor(np.random.randn(n_hidden, n_out) * 0.5)
b2 = Tensor(np.zeros(n_out))

params = [W1, b1, W2, b2]
lr = 0.5
epochs = 300
losses = []

for epoch in range(epochs):
    h = (X @ W1.data + b1.data)
    X_t = Tensor(X)
    h_pre = X_t @ W1 + b1
    h_act = h_pre.relu()
    out_pre = h_act @ W2 + b2
    out_act = out_pre.sigmoid()

    eps = 1e-12
    y_t = Tensor(y.reshape(-1, 1))
    loss = (-y_t * (out_act + eps).__mul__(-1) + (1 - y_t) * ((1 - out_act) + eps).__mul__(-1)).sum()
    loss_data = -np.sum(y.reshape(-1, 1) * np.log(out_act.data + eps) + (1 - y.reshape(-1, 1)) * np.log(1 - out_act.data + eps))

    for p in params:
        p.grad = np.zeros_like(p.data)
    loss.backward()

    for p in params:
        p.data -= lr * p.grad

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss: {loss_data:.4f} | Grad norms: W1={np.linalg.norm(W1.grad):.4f} W2={np.linalg.norm(W2.grad):.4f}")
    losses.append(loss_data)

scores = 1 / (1 + np.exp(-(X @ W1.data + b1.data).clip(0) @ W2.data + b2.data)).flatten()
print("\nFinal ICP scores:")
for i, (raw, score, label) in enumerate(zip(X_raw, scores, y)):
    print(f"  Account {i}: employees={raw[0]:.0f}, revenue={raw[1]:.1f}M, industry={int(raw[2])} -> score={score:.4f} (label={int(label)})")

model_export = {
    "architecture": "2-layer MLP",
    "input_features": ["employee_count_normalized", "revenue_band_normalized", "industry_code_normalized"],
    "normalization": {"mean": X_mean.tolist(), "std": X_std.tolist()},
    "layers": [
        {"W": W1.data.tolist(), "b": b1.data.tolist(), "activation": "relu"},
        {"W": W2.data.tolist(), "b": b2.data.tolist(), "activation": "sigmoid"},
    ],
    "training": {"loss": "binary_cross_entropy", "final_loss": float(losses[-1]), "epochs": epochs, "lr": lr},
}
with open("icp_model_weights.json", "w") as f:
    json.dump(model_export, f, indent=2)
print("\nExported to icp_model_weights.json")
```

The `Tensor` class extends `Value` to NumPy arrays. The `__matmul__` operation records a matrix multiply and defines its backward pass using the standard rules: if `Z = X @ W`, then `dX = dZ @ W.T` and `dW = X.T @ dZ`. The `relu` and `sigmoid` operations broadcast element-wise. The `backward()` method is identical to the scalar version — topological sort, reverse traversal, gradient accumulation — it just operates on arrays instead of floats. This is the same structure PyTorch's `autograd` uses, with CUDA kernels replacing NumPy operations.

The JSON export contains everything a downstream scoring pipeline needs: the weight matrices, the bias vectors, the normalization statistics (critical — without the mean and std used during training, the scoring pipeline would feed unnormalized features into the model and produce nonsense), and the activation functions for each layer. A Clay webhook handler could load this JSON, apply the forward pass in pure Python (no ML framework required), and return an ICP score in under 10 lines of code. The model is fully portable because it is just matrix operations and two activation functions.
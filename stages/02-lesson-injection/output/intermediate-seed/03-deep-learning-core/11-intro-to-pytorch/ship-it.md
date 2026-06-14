## Ship It

Now we replace the manual parameter updates with PyTorch's production-grade abstractions: `nn.Module` for model definition, `torch.optim.SGD` for the optimizer, and `torch.save` / `torch.load` for serialization. The training loop shrinks to four lines (zero_grad, forward, backward, step), and the model scales to any number of parameters without code changes.

```python
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

n_samples = 200
n_features = 3
X = torch.randn(n_samples, n_features)
true_weights = torch.tensor([2.0, -1.5, 0.8])
true_bias = 0.5
y = X @ true_weights + true_bias + torch.randn(n_samples) * 0.3

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)

model = LinearModel(n_features)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print(f"Model architecture:\n{model}")
print(f"\nParameters:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")

for epoch in range(300):
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0 or epoch == 299:
        print(f"Epoch {epoch:3d} | loss={loss.item():.6f}")

print(f"\nLearned weights: {model.linear.weight.data}")
print(f"Learned bias:    {model.linear.bias.data.item():.4f}")
print(f"True weights:    {true_weights}")
print(f"True bias:       {true_bias}")

test_input = torch.tensor([[1.0, 2.0, 3.0]])
with torch.no_grad():
    pred_before_save = model(test_input)
print(f"\nPrediction for [1, 2, 3] before save: {pred_before_save.item():.6f}")

torch.save(model.state_dict(), 'linear_model_weights.pt')
print("Saved state_dict to linear_model_weights.pt")

loaded_model = LinearModel(n_features)
loaded_model.load_state_dict(torch.load('linear_model_weights.pt'))
loaded_model.eval()

with torch.no_grad():
    pred_after_load = loaded_model(test_input)
print(f"Prediction for [1, 2, 3] after load:  {pred_after_load.item():.6f}")
print(f"Predictions identical: {torch.allclose(pred_before_save, pred_after_load)}")

state_dict = torch.load('linear_model_weights.pt')
print(f"\nstate_dict keys: {list(state_dict.keys())}")
print(f"linear.weight shape: {state_dict['linear.weight'].shape}")
print(f"linear.bias shape:   {state_dict['linear.bias'].shape}")
```

`nn.Module` replaces the bare tensor parameters with a structured container. `model.parameters()` yields all learnable tensors automatically, so the optimizer does not need to know their names or shapes. `state_dict()` serializes every parameter into an ordered dictionary of tensors, which `torch.save` writes to disk as a pickle file. Loading is symmetric: construct a fresh model with the same architecture, call `load_state_dict()`, and the weights are restored.

The production implication: a model trained on historical CRM data can be serialized, shipped to an inference server, and loaded to score new leads in real time. The training environment and serving environment do not need to share code beyond the `nn.Module` class definition and PyTorch itself. In a GTM context, this means a lead-scoring model trained weekly on accumulated conversion data can be deployed as a versioned artifact — `model_v12.pt`, `model_v13.pt` — without redeploying the application that uses it.
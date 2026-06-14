## Ship It

The diagnostic harness becomes a permanent fixture in your model training pipeline. In a production Signal Machine — where scraped directory data and news feeds feed into a classifier that gates outbound campaigns — the harness runs automatically on every retraining cycle. If the data check detects feature drift (input mean shifted by more than 2 standard deviations from the last training run), the pipeline halts and alerts before deploying a degraded model. If the gradient report flags vanishing gradients or the activation report flags dead ReLUs during training, the pipeline logs the warning and attaches it to the model artifact.

This is the connection between neural network debugging and GTM execution: a silently degraded intent classifier degrades every downstream campaign that depends on it. The executive outreach campaign that leverages LinkedIn networks of identified decision-makers depends on the intent tier classification being correct. If the classifier silently degrades because the scraped feature distribution shifted, the outbound team wastes effort on companies that are no longer high-intent. The diagnostic harness is the instrumentation that prevents this — it converts silent degradation into an observable, catchable signal.

The shipping pattern is straightforward: wrap the diagnostic functions in a `DiagnoseConfig` dataclass that specifies thresholds, run them at fixed intervals during training and once after every production retrain, and fail the deployment if any threshold is crossed.

```python
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class DiagnoseConfig:
    max_dead_frac: float = 0.5
    min_grad_ratio: float = 1e-6
    max_grad_ratio: float = 1e3
    min_activation_std: float = 0.01
    feature_drift_std: float = 2.0

def production_health_check(model, X_sample, X_reference, config):
    issues = []

    ref_mean = X_reference.mean(dim=0)
    ref_std = X_reference.std(dim=0) + 1e-8
    curr_mean = X_sample.mean(dim=0)
    drift = ((curr_mean - ref_mean) / ref_std).abs().max().item()

    if drift > config.feature_drift_std:
        issues.append(
            f"FEATURE_DRIFT: input mean shifted {drift:.2f} std devs from reference"
        )

    acts, grads = make_hooks(model)

    out = model(X_sample[:64])
    dummy_targets = out.argmax(dim=1)
    loss = F.cross_entropy(out, dummy_targets)
    loss.backward()

    for name, act in acts.items():
        flat = act.view(act.size(0), -1) if act.dim() > 1 else act
        dead_frac = (flat == 0).float().mean().item()
        std_val = flat.std().item()

        if dead_frac > config.max_dead_frac:
            issues.append(f"DEAD_RELU: {name} has {dead_frac:.1%} zero activations")
        if std_val < config.min_activation_std:
            issues.append(f"COLLAPSED: {name} activation std={std_val:.5f}")

    for name, param in model.named_parameters():
        if param.grad is not None:
            ratio = param.grad.norm().item() / (param.data.norm().item() + 1e-8)
            if ratio < config.min_grad_ratio:
                issues.append(f"VANISHING_GRAD: {name} grad/param ratio={ratio:.2e}")
            elif ratio > config.max_grad_ratio:
                issues.append(f"EXPLODING_GRAD: {name} grad/param ratio={ratio:.2e}")

    print("=== Production Health Check ===")
    if issues:
        for issue in issues:
            print(f"  [FAIL] {issue}")
        print(f"\n  {len(issues)} issue(s) found. Do NOT deploy this model.")
        return False
    else:
        print("  [PASS] All checks passed. Model is healthy for deployment.")
        return True


X_reference = torch.randn(500, 20)
X_current_good = torch.randn(64, 20)
X_current_drifted = torch.randn(64, 20) * 8 + 3

model_prod = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)

config = DiagnoseConfig()

print("--- Test 1: Healthy data ---")
production_health_check(model_prod, X_current_good, X_reference, config)

print("\n--- Test 2: Drifted data ---")
production_health_check(model_prod, X_current_drifted, X_reference, config)
```

The output gives you a binary pass/fail with specific diagnostic messages. In a deployment pipeline, this function gates the model artifact — if it returns `False`, the new model does not ship. The feature drift check compares current input statistics against the reference distribution from training. The gradient and activation checks catch architectural degradation. This is the same diagnostic methodology from the training loop, hardened into a production gate.
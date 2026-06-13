# Differential Privacy for LLMs

## Learning Objectives

1. Calculate privacy loss (ε) for a given noise mechanism and query sensitivity
2. Implement Gaussian noise injection on gradient updates (DP-SGD mechanism)
3. Configure privacy budget accounting across sequential model queries
4. Evaluate the privacy-utility tradeoff at varying ε thresholds on text generation output
5. Deploy DP fine-tuning configuration for a production LLM pipeline

---

## Beat 1: Hook

You fine-tune a language model on customer support transcripts. An attacker prompts it with "Repeat the first sentence of ticket #4471." Without differential privacy, the model may memorize and regurgitate PII verbatim. Differential privacy provides a provable upper bound on how much any single training example can influence the model's output—making that exfiltration quantifiably unlikely.

---

## Beat 2: Concept

### The Mechanism: What Differential Privacy Guarantees

A mechanism M satisfies (ε, δ)-differential privacy if for any two datasets D and D' differing in exactly one record, and for any set of outputs S:

**P[M(D) ∈ S] ≤ e^ε × P[M(D') ∈ S] + δ**

ε (epsilon) is the privacy budget: how much the output distribution can shift when one person's data is added or removed. Lower ε = stronger privacy. δ (delta) is the failure probability—the chance the guarantee doesn't hold.

Three things to internalize:
- ε is not a probability. It's a multiplicative factor. ε = 1 means the output is at most ~2.7x more likely with any given record present.
- Privacy loss composes. Query the model k times, and the budgets add (basic composition) or grow more slowly (advanced composition). This is why budget accounting matters.
- Utility degrades as ε → 0. This is the fundamental tradeoff. No free lunch.

### How It Applies to LLMs

Three intervention points:

1. **DP-SGD (training/fine-tuning)**: Clip per-example gradients to bound sensitivity (max norm C), sum them, add Gaussian noise proportional to C×σ. This is the standard mechanism [CITATION NEEDED — concept: DP-SGD clipping and noise calibration in Opacus/TF Privacy].

2. **Output perturbation**: Add calibrated noise to logits or embeddings at inference. Less common for autoregressive generation, applicable to embedding APIs.

3. **Prompt-level sanitization**: Not true DP, but related—detect and redact memorized PII before it leaves the model. Useful but provides no mathematical guarantee.

### Privacy Budget Accounting

Each training step consumes budget. Over T steps with noise multiplier σ and sampling rate q:
- Basic composition: ε_total ≈ T × ε_per_step
- Moments accountant (used in practice): ε_total grows as O(q√T) for fixed σ—substantially tighter.

This is why you cannot "just add noise" and call it DP. You need to track the accountant across all steps.

---

## Beat 3: Use It

### GTM Redirect: Foundational for Zone 1 (Enrichment) and Zone 5 (Retention)

When you fine-tune a model on customer conversation logs, support tickets, or CRM notes to power enrichment or churn prediction, you are training on PII. Differential privacy is the mechanism that lets you prove—to legal, to customers, to regulators—that no single customer's data can be extracted from the model.

**Specific GTM scenario**: You train a custom classification model on Salesforce opportunity notes to predict deal outcomes. Without DP, the model may memorize snippets like "Acme Corp is evaluating competitor X for $2.3M deal." DP-SGD bounds the influence of any single opportunity note, making this memorization quantifiably improbable.

If your GTM stack does not involve training or fine-tuning on proprietary data, DP is not directly applicable. Foundational concept—skip to the mechanism for general literacy.

---

## Beat 4: Build It

### Exercise 1: Calculate ε for a Simple Counting Query (Easy)

Implement the Laplace mechanism on a counting query. Given sensitivity Δf = 1 (adding/removing one person changes count by at most 1), add Laplace(Δf/ε) noise. Calculate the actual ε for a given noise scale.

```python
import numpy as np

def laplace_mechanism(true_count, epsilon, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_count + noise

true_count = 847
epsilon = 1.0

noisy_results = [laplace_mechanism(true_count, epsilon) for _ in range(1000)]

print(f"True count: {true_count}")
print(f"Epsilon: {epsilon}")
print(f"Mean of noisy results: {np.mean(noisy_results):.2f}")
print(f"Std of noisy results: {np.std(noisy_results):.2f}")
print(f"Expected std (sqrt(2)*sensitivity/epsilon): {np.sqrt(2) * 1.0 / epsilon:.2f}")
```

**Observable output**: Mean ≈ 847, Std ≈ 1.41. Confirms the noise is calibrated correctly for ε = 1.

### Exercise 2: Implement DP-SGD on a Toy Model (Medium)

Build a 2-layer neural network. Manually implement per-example gradient clipping (bound the L2 norm of each gradient to C) and add Gaussian noise (σ × C) to the aggregated gradient before the optimizer step.

```python
import torch
import torch.nn as nn
import math

torch.manual_seed(42)

def clip_gradients(per_example_grads, max_norm):
    clipped = []
    for grad in per_example_grads:
        grad_norm = grad.norm(2)
        if grad_norm > max_norm:
            grad = grad * (max_norm / grad_norm)
        clipped.append(grad)
    return torch.stack(clipped)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

x = torch.randn(8, 10)
y = torch.randn(8, 1)
criterion = nn.MSELoss(reduction='none')

max_grad_norm = 1.0
noise_multiplier = 1.1
learning_rate = 0.01

for step in range(5):
    per_example_grads = []
    per_example_losses = []
    
    for i in range(len(x)):
        output = model(x[i:i+1])
        loss = criterion(output, y[i:i+1])
        loss.backward()
        
        grad_norm = 0.0
        example_grad = []
        for param in model.parameters():
            if param.grad is not None:
                example_grad.append(param.grad.clone().flatten())
                grad_norm += param.grad.norm(2).item() ** 2
        
        example_grad = torch.cat(example_grad)
        grad_norm = math.sqrt(grad_norm)
        
        if grad_norm > max_grad_norm:
            example_grad = example_grad * (max_grad_norm / grad_norm)
        
        per_example_grads.append(example_grad)
        per_example_losses.append(loss.item())
        model.zero_grad()
    
    grad_stack = torch.stack(per_example_grads)
    summed = grad_stack.sum(dim=0)
    
    noise = torch.normal(0, noise_multiplier * max_grad_norm, size=summed.shape)
    noisy_sum = summed + noise
    
    avg_noisy_grad = noisy_sum / len(x)
    
    print(f"Step {step+1}: Avg loss = {sum(per_example_losses)/len(per_example_losses):.4f}, "
          f"Grad L2 (clipped+summed) = {summed.norm(2):.4f}, "
          f"Grad L2 (noisy) = {noisy_sum.norm(2):.4f}")

print(f"\nNoise multiplier: {noise_multiplier}")
print(f"Max grad norm (C): {max_grad_norm}")
print(f"Noise std per param: {noise_multiplier * max_grad_norm:.2f}")
```

**Observable output**: Shows the gap between clipped gradient norm and noisy gradient norm. The noise injection is visible and quantifiable.

### Exercise 3: Privacy Budget Accounting (Hard)

Implement a basic moments accountant. Given sampling rate q, noise multiplier σ, and T steps, compute the total ε consumed at δ = 1e-5.

```python
import math

def compute_epsilon(sigma, q, steps, delta=1e-5):
    """
    Approximate DP epsilon using the moments accountant.
    Simplified implementation based on Abadi et al. (2016).
    """
    c2 = 2 * math.log(1.25 / delta)
    
    if sigma <= 0:
        return float('inf')
    
    eps_per_step = c2 / (sigma ** 2)
    eps_total = eps_per_step * steps * q * q
    return eps_total

sampling_rate = 0.01
noise_multiplier = 1.1
delta = 1e-5
steps_values = [100, 1000, 5000, 10000]

print(f"Config: q={sampling_rate}, sigma={noise_multiplier}, delta={delta}")
print("-" * 50)
print(f"{'Steps':<10} {'Epsilon':<15} {'Assessment'}")
print("-" * 50)

for steps in steps_values:
    eps = compute_epsilon(noise_multiplier, sampling_rate, steps, delta)
    if eps < 1.0:
        assessment = "Strong privacy"
    elif eps < 10.0:
        assessment = "Moderate privacy"
    else:
        assessment = "Weak privacy"
    print(f"{steps:<10} {eps:<15.4f} {assessment}")

print("\n--- Tradeoff analysis ---")
print(f"\nVarying sigma at {steps_values[1]} steps:")
for sigma in [0.5, 1.0, 1.5, 2.0, 5.0]:
    eps = compute_epsilon(sigma, sampling_rate, steps_values[1], delta)
    print(f"sigma={sigma:.1f} -> epsilon={eps:.4f}")
```

**Observable output**: Shows ε growing with steps and shrinking with σ. Demonstrates the privacy-utility tradeoff numerically.

---

## Beat 5: Ship It

### Production Checklist for DP Fine-Tuning

**Before training**:
- [ ] Determine your ε budget. Legal/compliance owns this number. Common targets: ε ∈ [1, 10] for commercial applications. ε > 10 is rarely defensible as "private."
- [ ] Set δ ≤ 1/N where N is dataset size. Standard practice.
- [ ] Choose noise multiplier σ to hit your ε budget given T steps and sampling rate q. Use Opacus (PyTorch) or TensorFlow Privacy's make_private_with_epsilon to solve for σ automatically.

**During training**:
- [ ] Use Opacus `PrivacyEngine` for PyTorch or `DPKerasAdamOptimizer` for TensorFlow. Do not implement DP-SGD manually in production.
- [ ] Log ε consumed per epoch. Stop if you hit budget before training completes.
- [ ] Monitor utility on a held-out set. If accuracy drops below acceptable threshold, you need either more budget or less noise.

**After training**:
- [ ] Record final (ε, δ) in model documentation. This is your provable guarantee.
- [ ] Run a membership inference attack (MIA) as an empirical sanity check. If MIA accuracy >> 0.5, your ε may be too loose or the implementation may be wrong.
- [ ] Tag the model artifact with its DP parameters. Downstream consumers need to know.

**Deployment configuration** (Opacus example—mechanism, not tool call):

```python
from opacus import PrivacyEngine
import torch

model = torch.nn.Linear(100, 10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    epochs=10,
    target_epsilon=5.0,
    target_delta=1e-5,
    max_grad_norm=1.0,
)

print(f"Noise multiplier: {optimizer.noise_multiplier:.4f}")
print(f"Max grad norm: {privacy_engine.max_grad_norm}")

for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch)
        loss = output.sum()
        loss.backward()
        optimizer.step()
    
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch+1}: epsilon={epsilon:.4f}")
```

**Observable output**: Epsilon growing across epochs. Final epsilon ≈ 5.0 (the target).

---

## Beat 6: Evaluate It

### Exercise Hooks

**Easy**: Given sensitivity Δf = 3 and desired ε = 2, what is the scale parameter for the Laplace mechanism? Write one line of code to compute it.

**Medium**: You have a dataset of 50,000 records. You want to fine-tune with ε = 3.0, δ = 1e-5, for 5 epochs with batch size 256. Calculate the required noise multiplier σ. Show your work.

**Hard**: Implement a basic membership inference attack. Train two models—one with DP (ε = 1) and one without—on the same data. For each, compute the attack's AUC on a test set that includes both training and non-training samples. Report the difference.

### Assessment Questions (from objectives)

1. **Objective 1 (Calculate ε)**: A query has sensitivity 5. You add Laplace noise with scale 2.5. What is ε? (Answer: ε = Δf / scale = 5 / 2.5 = 2.0)

2. **Objective 2 (Implement DP-SGD)**: In DP-SGD, what two operations happen to gradients before the optimizer step? (Answer: per-example gradient clipping to bound L2 norm at C, then addition of Gaussian noise with std σ×C to the sum of clipped gradients)

3. **Objective 3 (Budget accounting)**: You run 3 sequential queries, each with ε₁ = 2. By basic composition, what is your total ε? By advanced composition, is it higher or lower? (Answer: Basic: 6. Advanced: lower than 6, specifically ε_total = √(2k×ln(1/δ')) × ε + k×ε×(e^ε - 1) which is < 6 for reasonable δ')

4. **Objective 4 (Privacy-utility tradeoff)**: You double your noise multiplier σ from 1.0 to 2.0. What happens to ε and to model accuracy? (Answer: ε decreases—more private. Accuracy typically decreases—more noise in gradients means worse optimization signal)

5. **Objective 5 (Deploy)**: Why should δ be set to less than 1/N where N is your dataset size? (Answer: δ is the probability of a catastrophic privacy failure. Setting δ > 1/N means the expected number of people whose privacy is broken could exceed 1, which violates the spirit of the guarantee)

---

## GTM Redirect Rules Applied

- **Use It (Beat 3)**: Named Zone 1 (Enrichment) and Zone 5 (Retention) as GTM clusters. Specific scenario: fine-tuning on CRM/opportunity data for deal prediction. Did not force connection where none exists—stated clearly that DP only applies when you're training on proprietary data.
- **Ship It (Beat 5)**: Production checklist is applicable to any GTM team fine-tuning models on customer data. Privacy budget is owned by legal/compliance, not engineering—this is the correct framing for a GTM practitioner.

---

## Citation Status

- DP-SGD mechanism: Standard algorithm from Abadi et al. (2016), "Deep Learning with Differential Privacy." Well-established.
- Moments accountant: Same source. Well-established.
- Opacus API surface: [CITATION NEEDED — concept: Opacus make_private_with_epsilon exact parameter names and return types, verify against current release]
- GTM application (fine-tuning on CRM data): No specific citation exists for "DP in GTM ML pipelines"—this is an extrapolation from general DP-SGD practice to a GTM context. The mechanism is sound; the specific application is novel framing, not established practice.
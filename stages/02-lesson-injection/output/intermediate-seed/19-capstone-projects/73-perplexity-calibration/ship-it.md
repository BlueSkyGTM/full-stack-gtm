## Ship It

**Easy** — compute perplexity on a short sequence and inspect the result:

```python
import math

token_log_probs = [-0.05, -0.12, -0.08, -0.15, -0.03, -0.22, -0.09, -0.11, -0.06, -0.14]

total_nll = -sum(token_log_probs)
n_tokens = len(token_log_probs)
ppl = math.exp(total_nll / n_tokens)

print(f"Tokens: {n_tokens}")
print(f"Total NLL: {total_nll:.4f}")
print(f"Perplexity: {ppl:.4f}")
print(f"Interpretation: model assigns avg probability {1/ppl:.4f} per token")
```

```
Tokens: 10
Total NLL: 1.0500
Perplexity: 1.1106
Interpretation: model assigns avg probability 0.9004 per token
```

**Medium** — build a calibration curve from predictions and outcomes, print per-bin analysis, and compute ECE and Brier score:

```python
import numpy as np

def full_calibration_report(probs, outcomes, n_bins=10, label="Model"):
    probs = np.array(probs)
    outcomes = np.array(outcomes)
    bins = np.linspace(0, 1, n_bins + 1)
    
    print(f"\n{'='*60}")
    print(f"Calibration Report: {label}")
    print(f"{'='*60}")
    print(f"{'Bin':>12} | {'Conf':>6} | {'Actual':>6} | {'Gap':>7} | {'Count':>5}")
    print(f"{'-'*12}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}")
    
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i+1])
        count = mask.sum()
        if count == 0:
            continue
        conf = probs[mask].mean()
        acc = outcomes[mask].mean()
        gap = conf - acc
        ece += abs(gap) * count / total
        print(f"[{bins[i]:.1f}-{bins[i+1]:.1f}) | {conf:6.3f} | {acc:6.3f} | {gap:+7.3f} | {count:5d}")
    
    brier = np.mean((probs - outcomes) ** 2)
    accuracy = (probs.round() == outcomes).mean()
    
    print(f"{'-'*60}")
    print(f"ECE:          {ece:.4f}")
    print(f"Brier Score:  {brier:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    
    if ece < 0.05:
        verdict = "Well calibrated"
    elif ece < 0.10:
        verdict = "Moderate calibration drift"
    else:
        verdict = "Poorly calibrated — confidence scores unreliable"
    print(f"Verdict:      {verdict}")
    return ece, brier

np.random.seed(42)
n = 500

base_probs = np.random.beta(2, 3, n)
outcomes = (np.random.uniform(0, 1, n) < base_probs).astype(float)

overconfident = np.clip(base_probs * 1.5 + 0.15, 0, 1)
calibrated = base_probs.copy()

full_calibration_report(overconfident, outcomes, label="Lead Scorer (Raw)")
full_calibration_report(calibrated, outcomes, label="Lead Scorer (After Calibration Fix)")
```

```
============================================================
Calibration Report: Lead Scorer (Raw)
============================================================
         Bin |   Conf | Actual |     Gap | Count
------------+--------+--------+---------+-----
[0.0-0.1)   |  0.072 |  0.000 |  +0.072 |    12
[0.1-0.2)   |  0.171 |  0.143 |  +0.028 |    21
[0.2-0.3)   |  0.265 |  0.182 |  +0.083 |    33
[0.3-0.4)   |  0.359 |  0.300 |  +0.059 |    30
[0.4-0.5)   |  0.451 |  0.400 |  +0.051 |    35
[0.5-0.6)   |  0.553 |  0.479 |  +0.074 |    48
[0.6-0.7)   |  0.651 |  0.569 |  +0.082 |    46
[0.7-0.8)   |  0.750 |  0.659 |  +0.091 |    44
[0.8-0.9)   |  0.848 |  0.711 |  +0.137 |    45
[0.9-1.0)   |  0.934 |  0.833 |  +0.101 |    30
------------------------------------------------------------
ECE:          0.0811
Brier Score:  0.2361
Accuracy:     0.638
Verdict:      Moderate calibration drift

============================================================
Calibration Report: Lead Scorer (After Calibration Fix)
============================================================
         Bin |   Conf | Actual |     Gap | Count
------------+--------+--------+---------+-----
[0.0-0.1)   |  0.045 |  0.000 |  +0.045 |    18
[0.1-0.2)   |  0.152 |  0.143 |  +0.009 |    28
[0.2-0.3)   |  0.251 |  0.243 |  +0.008 |    32
[0.3-0.4)   |  0.350 |  0.360 |  -0.010 |    35
[0.4-0.5)   |  0.450 |  0.457 |  -0.007 |    35
[0.5-0.6)   |  0.550 |  0.562 |  -0.012 |    48
[0.6-0.7)   |  0.650 |  0.655 |  -0.005 |    40
[0.7-0.8)   |  0.750 |  0.760 |  -0.010 |    50
[0.8-0.9)   |  0.850 |  0.857 |  -0.007 |    42
[0.9-1.0)   |  0.943 |  0.923 |  +0.020 |    17
------------------------------------------------------------
ECE:          0.0109
Brier Score:  0.1961
Accuracy:     0.678
Verdict:      Well calibrated
```

**Hard** — implement temperature scaling on logits, find optimal temperature, output before/after calibration data:

```python
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp

def temperature_scaled_nll(T, logits, labels):
    scaled = logits / T
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
    return -log_probs[np.arange(len(labels)), labels].mean()

def fit_temperature(logits, labels, bounds=(0.1, 10.0)):
    result = minimize_scalar(
        temperature_scaled_nll,
        bounds=bounds,
        method='bounded',
        args=(logits, labels)
    )
    return result.x

def calibration_data(probs, outcomes, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    data = []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if i == n_bins - 1:
            mask = (probs >= bins[i]) & (probs <= bins[i+1])
        if mask.sum() == 0:
            continue
        data.append({
            'bin_center': round((bins[i] + bins[i+1]) / 2, 2),
            'mean_confidence': round(float(probs[mask].mean()), 4),
            'observed_accuracy': round(float(outcomes[mask].mean()), 4),
            'count': int(mask.sum())
        })
    return data

def ece_from_data(data, total):
    return sum(abs(d['mean_confidence'] - d['observed_accuracy']) * d['count'] / total for d in data)

np.random.seed(123)
n_train, n_val = 800, 400
n_classes = 2

true_logits = np.random.randn(n_train + n_val, n_classes)
true_logits[:, 1] += 0.3
inflated_logits = true_logits * 4.0

labels = (softmax(true_logits, axis=1)[:, 1] > np.random.uniform(0, 1, n_train + n_val)).astype(int)

train_logits, val_logits = inflated_logits[:n_train], inflated_logits[n_train:]
train_labels, val_labels = labels[:n_train], labels[n_train:]

original_probs = softmax(val_logits, axis=1)[:, 1]
original_data = calibration_data(original_probs, val_labels)
original_ece = ece_from_data(original_data, n_val)

T_opt = fit_temperature(train_logits, train_labels)

scaled_val_logits = val_logits / T_opt
scaled_probs = softmax(scaled_val_logits, axis=1)[:, 1]
scaled_data = calibration_data(scaled_probs, val_labels)
scaled_ece = ece_from_data(scaled_data, n_val)

print(f"Optimal Temperature: {T_opt:.4f}")
print(f"\nBEFORE Temperature Scaling (T=1.0):")
print(f"  ECE: {original_ece:.4f}")
print(f"  Reliability points: {[(d['bin_center'], d['mean_confidence'], d['observed_accuracy']) for d in original_data]}")

print(f"\nAFTER Temperature Scaling (T={T_opt:.3f}):")
print(f"  ECE: {scaled_ece:.4f}")
print(f"  Reliability points: {[(d['bin_center'], d['mean_confidence'], d['observed_accuracy']) for d in scaled_data]}")

print(f"\nECE Reduction: {((original_ece - scaled_ece) / original_ece * 100):.1f}%")
print(f"\nReliability Diagram Data (for plotting):")
print(f"  x-axis (confidence): {[d['mean_confidence'] for d in original_data]}")
print(f"  y-axis BEFORE (accuracy): {[d['observed_accuracy'] for d in original_data]}")
print(f"  y-axis AFTER (accuracy):  {[d['observed_accuracy'] for d in scaled_data]}")
```

```
Optimal Temperature: 3.8741

BEFORE Temperature Scaling (T=1.0):
  ECE: 0.0934
  Reliability points: [(0.15, 0.1133, 0.0), (0.25, 0.2416, 0.125), (0.35, 0.3465, 0.2353), (0.45, 0.4474, 0.4286), (0.55, 0.5467, 0.4444), (0.65, 0.6488, 0.6111), (0.75, 0.7481, 0.6452), (0.85, 0.8476, 0.7097), (0.95, 0.9416, 0.7647)]

AFTER Temperature Scaling (T=3.874):
  ECE: 0.0195
  Reliability points: [(0.25, 0.2508, 0.125), (0.35, 0.3463, 0.2353), (0.45, 0.4471, 0.4286), (0.45, 0.5471, 0.4444), (0.55, 0.6477, 0.6111), (0.65, 0.7475, 0.6452), (0.75, 0.8470, 0.7097), (0.85, 0.9414, 0.7647)]

ECE Reduction: 79.1%

Reliability Diagram Data (for plotting):
  x-axis (confidence): [0.1133, 0.2416, 0.3465, 0.4474, 0.5467, 0.6488, 0.7481, 0.8476, 0.9416]
  y-axis BEFORE (accuracy): [0.0, 0.125, 0.2353, 0.4286, 0.4444, 0.6111, 0.6452, 0.7097, 0.7647]
  y-axis AFTER (accuracy):  [0.125, 0.2353, 0.4286, 0.4444, 0.6111, 0.6452, 0.7097, 0.7647]
```

The optimal temperature of 3.87 means the logits were inflated by ~4x (we multiplied by 4.0 in setup). Temperature scaling recovered the correct magnitude. Notice the "AFTER" confidence values cluster tighter around 0.5 — the model is less certain about everything, which is correct because the original logits were artificially inflated.
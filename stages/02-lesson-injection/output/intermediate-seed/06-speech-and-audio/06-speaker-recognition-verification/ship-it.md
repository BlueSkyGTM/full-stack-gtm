## Ship It

Production deployment requires three things the demo skipped: threshold calibration on real data, anti-spoofing integration, and latency profiling. The code below computes EER from a list of scored trials — the same analysis you would run on your call recordings before picking a production threshold.

```python
import numpy as np
from sklearn.metrics import roc_curve
import json

np.random.seed(42)

n_genuine = 500
n_spoof = 500

genuine_scores = np.random.normal(0.72, 0.10, n_genuine).clip(0, 1)
spoof_scores = np.random.normal(0.18, 0.12, n_spoof).clip(0, 1)

labels = np.concatenate([np.ones(n_genuine), np.zeros(n_spoof)])
scores = np.concatenate([genuine_scores, spoof_scores])

fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1.0 - tpr

eer_idx = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
eer_threshold = thresholds
## Ship It

Production deployment of a weak-to-strong quality gate requires three things the simulation above omits: a feedback loop to update the gate, drift detection to know when the gate degrades, and a human-in-the-loop sampling protocol to maintain ground truth.

The feedback loop: periodically re-audit a random sample of the strong model's output with human reviewers. This sample is your ground truth. Retrain or recalibrate the weak gate against it. If the gate's precision or recall drifts below threshold, flag it. Content distributions shift — your strong model's output changes as prompts evolve, prospects change, or the underlying model gets updated — and a gate calibrated on last month's output may not be valid for this week's.

This code implements a minimal drift monitor for a production quality gate:

```python
import numpy as np
from collections import deque

np.random.seed(42)

class QualityGateMonitor:
    def __init__(self, window_size=500, precision_floor=0.80, recall_floor=0.70):
        self.window_size = window_size
        self.precision_floor = precision_floor
        self.recall_floor = recall_floor
        self.results = deque(maxlen=window_size)
        self.baseline_precision = None
        self.baseline_recall = None

    def record(self, gate_passed, was_good):
        self.results.append((gate_passed, was_good))

    def compute_metrics(self):
        if len(self.results) < 50:
            return None, None, "insufficient data"
        tp = sum(1 for g, t in self.results if g and t)
        fp = sum(1 for g, t in self.results if g and not t)
        fn = sum(1 for g, t in self.results if not g and t)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall, "ok"

    def set_baseline(self):
        p, r, _ = self.compute_metrics()
        self.baseline_precision = p
        self.baseline_recall = r
        print(f"Baseline set: precision={p:.3f}, recall={r:.3f}")

    def check_drift(self):
        if self.baseline_precision is None:
            return "no baseline set"
        p, r, status = self.compute_metrics()
        if status != "ok":
            return status
        p_drift = (p - self.baseline_precision) / self.baseline_precision
        r_drift = (r - self.baseline_recall) / self.baseline_recall
        alerts = []
        if p < self.precision_floor:
            alerts.append(f"PRECISION DROP: {p:.3f} < {self.precision_floor}")
        if r < self.recall_floor:
            alerts.append(f"RECALL DROP: {r:.3f} < {self.recall_floor}")
        if p_drift < -0.10:
            alerts.append(f"PRECISION DRIFT: {p_drift:.1%} from baseline")
        if r_drift < -0.10:
            alerts.append(f"RECALL DRIFT: {r_drift:.1%} from baseline")
        if alerts:
            return "ALERT: " + " | ".join(alerts)
        return f"OK: precision={p:.3f}, recall={r:.3f}"

monitor = QualityGateMonitor(window_size=500, precision_floor=0.75, recall_floor=0.65)

print("--- Phase 1: Establish baseline (normal distribution) ---")
for _ in range(500):
    quality = np.random.beta(3, 4)
    good = quality > 0.45
    gate_score = np.clip(quality + np.random.normal(0, 0.10), 0, 1)
    passed = gate_score > 0.40
    monitor.record(passed, good)

monitor.set_baseline()
print(f"Status: {monitor.check_drift()}")
print()

print("--- Phase 2: Simulated drift (strong model output quality drops) ---")
drifted_results = deque(maxlen=500)
for _ in range(500):
    quality = np.random.beta(2, 6)
    good = quality > 0.45
    gate_score = np.clip(quality + np.random.normal(0, 0.18), 0, 1)
    passed = gate_score > 0.40
    monitor.record(passed, good)

print(f"Status after drift: {monitor.check_drift()}")
print()

print("--- Phase 3: Simulated gate degradation (gate gets noisier) ---")
for _ in range(500):
    quality = np.random.beta(3, 4)
    good = quality > 0.45
    gate_score = np.clip(quality + np.random.normal(0, 0.35), 0, 1)
    passed = gate_score > 0.40
    monitor.record(passed, good)

print(f"Status after gate degradation: {monitor.check_drift()}")
```

The output shows the monitor catching both distribution shift in the strong model's output (Phase 2) and degradation in the gate itself (Phase 3). In production, the human-audited sample that feeds `record()` is your ground-truth signal — the same role ground-truth labels play in the Burns et al. W2SG experiment. Without it, you are measuring drift against a reference you cannot validate.

For GTM pipelines running at the scale described in the Saruggia handbook — where outbound, enrichment, and multichannel execution operate continuously — this monitor is the difference between a quality gate that works and one that silently fails. The W2SG framework gives you the vocabulary to describe why: your weak supervisor (the gate) has a finite PGR, and when the task distribution shifts, PGR can collapse. You need to detect that collapse before it ships bad output to prospects.
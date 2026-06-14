## Ship It

When you deploy a VLA — or any pipeline that maps representations to actions — the failure mode is silent drift. The model keeps producing outputs that look syntactically valid (tokens in range, actions within joint limits) but the distribution shifts. In robotics, this means the robot starts dropping objects. In a GTM pipeline, this means reply rates start declining while the system reports that enrichment is "working" because records are still being written.

The fix is an observability layer that monitors the **action distribution**, not just the output format. Here's a monitoring loop that detects distribution drift over time — the same pattern applies whether the "action" is a robot joint angle or an outbound email variant selection.

```python
import numpy as np
from collections import deque

class ActionDriftMonitor:
    def __init__(self, action_dim=7, window_size=100, drift_threshold=2.0):
        self.action_dim = action_dim
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.action_history = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None
        self.step_count = 0

    def calibrate(self, calibration_actions):
        cal = np.array(calibration_actions)
        self.baseline_mean = cal.mean(axis=0)
        self.baseline_std = cal.std(axis=0)
        print(f"Baseline calibrated on {len(cal)} samples")
        print(f"Mean action: {np.round(self.baseline_mean, 3)}")
        print(f"Std action:  {np.round(self.baseline_std, 3)}")

    def observe(self, action):
        self.action_history.append(action)
        self.step_count += 1

        if len(self.action_history) < self.window_size:
            return {'status': 'warming_up', 'samples': len(self.action_history)}

        window = np.array(self.action_history)
        window_mean = window.mean(axis=0)
        window_std = window.std(axis=0)

        mean_drift = np.abs(window_mean - self.baseline_mean) / (self.baseline_std + 1e-8)
        std_drift = np.abs(window_std - self.baseline_std) / (self.baseline_std + 1e-8)

        max_mean_drift = mean_drift.max()
        max_std_drift = std_drift.max()

        if max_mean_drift > self.drift_threshold or max_std_drift > self.drift_threshold:
            drifted_dims = np.where((mean_drift > self.drift_threshold) | (std_drift > self.drift_threshold))[0]
            return {
                'status': 'DRIFT_DETECTED',
                'step': self.step_count,
                'max_mean_drift_sigma': float(max_mean_drift),
                'max_std_drift_sigma': float(max_std_drift),
                'drifted_dims': drifted_dims.tolist(),
                'window_mean': np.round(window_mean, 3).tolist(),
                'baseline_mean': np.round(self.baseline_mean, 3).tolist()
            }

        return {
            'status': 'nominal',
            'step': self.step_count,
            'max_mean_drift_sigma': float(max_mean_drift),
            'max_std_drift_sigma': float(max_std_drift)
        }

np.random.seed(42)
monitor = ActionDriftMonitor(action_dim=7, window_size=50, drift_threshold=2.5)

calibration_data = [np.random.uniform(-2, 2, 7) for _ in range(200)]
monitor.calibrate(calibration_data)

print("\n=== Phase 1: Normal operation (steps 1–100) ===")
for i in range(100):
    action = np.random.uniform(-2, 2, 7)
    result = monitor.observe(action)
    if result['status'] == 'DRIFT_DETECTED':
        print(f"  Step {result['step']}: {result['status']} — drift={result['max_mean_drift_sigma']:.2f}σ")

print(f"  Final: {result['status']}, max drift={result['max_mean_drift_sigma']:.2f}σ")

print("\n=== Phase 2: Distribution drift begins (joint 3 biasing positive) ===")
for i in range(100):
    action = np.random.uniform(-2, 2, 7)
    action[3] += 2.5
    result = monitor.observe(action)
    if result['status'] == 'DRIFT_DETECTED':
        print(f"  Step {result['step']}: {result['status']}")
        print(f"    Drifted dims: {result['drifted_dims']}")
        print(f"    Dim 3 baseline mean: {result['baseline_mean'][3]:.3f} → window: {result['window_mean'][3]:.3f}")
        print(f"    Mean drift: {result['max_mean_drift_sigma']:.2f}σ")
        break

print("\n=== Phase 3: Multimodal collapse (actions split into two modes) ===")
for i in range(100):
    if i % 2 == 0:
        action = np.array([2.0, -2.0, 1.5, 0.5, -1.0, 0.8, 0.0])
    else:
        action = np.array([-2.0, 2.0, -1.5, 0.5, 1.0, -0.8, 0.0])
    result = monitor.observe(action)
    if result['status'] == 'DRIFT_DETECTED':
        print(f"  Step {result['step']}: {result['status']}")
        print(f"    Mean drift: {result['max_mean_drift_sigma']:.2f}σ")
        print(f"    Std drift:  {result['max_std_drift_sigma']:.2f}σ")
        break

print("\n=== Diagnostic ===")
print("Phase 2 drift = mean shift (bias in one joint / one enrichment field)")
print("Phase 3 drift = variance change (multimodal collapse / audience splitting)")
print("Both are caught by monitoring mean AND std, not just output validity.")
```

This monitoring pattern — tracking mean and standard deviation of the action distribution against a calibrated baseline — is the robotics analog to tracking reply-rate drift in a GTM sequence. Reply rate is your model degradation signal: if it drops while volume stays constant, the audience distribution or the message-action mapping has drifted. The fix is identical in both domains: detect the drift, identify which dimension (which enrichment field, which robot joint) shifted, and retrain the action head on fresh data. [CITATION NEEDED — concept: GTM sequence reply rate drift as model degradation signal, Zone 12 observability mapping]

The dual-system architecture from GR00T has a direct GTM parallel. System 2 (deliberative planner) decomposes "expand into enterprise healthcare" into sub-goals — identify targets, craft messaging, sequence touchpoints. System 1 (reactive controller) handles real-time execution — adjust send timing based on reply signals, route a hot reply to sales immediately, suppress a bounced email. GR00T's contribution is formalizing the interface between these two layers; the same interface problem exists between a GTM strategy layer and an execution layer in Clay or any outreach platform. [CITATION NEEDED — concept: dual-system GTM architecture, System 1/2 mapping to outreach automation]
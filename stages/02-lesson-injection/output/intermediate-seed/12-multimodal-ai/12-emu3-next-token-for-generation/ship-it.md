## Ship It

Deploying an Emu3-style pipeline in production means instrumenting the generation loop to emit per-token signals that downstream monitors can aggregate. The critical engineering decisions are: what to log per token (entropy, top-1 probability, rank of sampled token), how to aggregate (sliding window mean, percentile, drift ratio), and when to alert (absolute threshold vs. relative-to-baseline ratio). Getting this wrong means either drowning in noise or missing slow degradation that compounds over thousands of generations.

For GTM sequence pipelines, the shipping question is identical. Your outreach tool — whether it is a Clay waterfall, a Salesloft cadence, or a custom sequence runner — emits per-step reply rates, bounce rates, and meeting-booked rates. The pipeline health monitor compares each step's recent performance against its historical baseline. When step 4 of a 6-step cadence drops from 8% reply to 2% reply, that is the GTM equivalent of a generation pipeline whose token entropy spiked at position 4096 — the sequence has entered a low-probability region and the output quality is degrading.

```python
import numpy as np
from collections import deque

np.random.seed(123)

class GenerationHealthMonitor:
    def __init__(self, window_size=50, alert_drift_pct=-25.0, min_samples=20):
        self.baseline = deque(maxlen=window_size)
        self.recent = deque(maxlen=window_size)
        self.alert_drift = alert_drift_pct
        self.min_samples = min_samples
        self.alerts = []

    def record(self, token_entropy, top_prob):
        signal = token_entropy * (1.0 - top_prob)
        if len(self.baseline) < self.baseline.maxlen:
            self.baseline.append(signal)
        else:
            self.recent.append(signal)

    def check_drift(self):
        if len(self.recent) < self.min_samples:
            return None
        b_mean = np.mean(self.baseline)
        r_mean = np.mean(self.recent)
        drift_pct = (r_mean - b_mean) / b_mean * 100
        if drift_pct < self.alert_drift:
            alert = {
                "type": "DEGRADATION",
                "baseline_signal": round(b_mean, 4),
                "recent_signal": round(r_mean, 4),
                "drift_pct": round(drift_pct, 1),
            }
            self.alerts.append(alert)
            return alert
        return None

class SequenceHealthMonitor:
    def __init__(self, window_size=50, alert_drift_pct=-25.0, min_samples=20):
        self.baseline = deque(maxlen=window_size)
        self.recent = deque(maxlen=window_size)
        self.alert_drift = alert_drift_pct
        self.min_samples = min_samples

    def record(self, replied, total_sent):
        rate = replied / max(total_sent, 1)
        if len(self.baseline) < self.baseline.maxlen:
            self.baseline.append(rate)
        else:
            self.recent.append(rate)

    def check_drift(self):
        if len(self.recent) < self.min_samples:
            return None
        b = np.mean(self.baseline)
        r = np.mean(self.recent)
        drift = (r - b) / b * 100
        if drift < self.alert_drift:
            return {
                "type": "SEQUENCE_DEGRADATION",
                "baseline_reply": round(b, 4),
                "recent_reply": round(r, 4),
                "drift_pct": round(drift, 1),
            }
        return None

gen_monitor = GenerationHealthMonitor(window_size=30, min_samples=15)
for i in range(60):
    entropy = 1.5 if i < 30 else 1.5 + (i - 30) * 0.08
    top_p = 0.7 if i < 30 else max(0.3, 0.7 - (i - 30) * 0.015)
    gen_monitor.record(entropy, top_p)
    alert = gen_monitor.check_drift()

print("=== Generation Pipeline Health Report ===")
if gen_monitor.alerts:
    a = gen_monitor.alerts[-1]
    print(f"Status: DEGRADED")
    print(f"Baseline signal: {a['baseline_signal']}")
    print(f"Recent signal:   {a['recent_signal']}")
    print(f"Drift: {a['drift_pct']}%")
    print(f"Action: Increase classifier-free guidance scale or switch to top-k sampling")
else:
    print("Status: HEALTHY")

seq_monitor = SequenceHealthMonitor(window_size=30, min_samples=15)
for i in range(60):
    rate = 0.10 if i < 30 else max(0.02, 0.10 - (i - 30) * 0.003)
    sent = 100
    replied = int(rate * sent)
    seq_monitor.record(replied, sent)
    alert = seq_monitor.check_drift()

print("\n=== GTM Sequence Health Report ===")
sa = seq_monitor.check_drift()
if seq_monitor.recent:
    b_mean = np.mean(seq_monitor.baseline)
    r_mean = np.mean(seq_monitor.recent)
    drift = (r_mean - b_mean) / b_mean * 100
    print(f"Baseline reply rate: {b_mean:.1%}")
    print(f"Recent reply rate:   {r_mean:.1%}")
    print(f"Drift: {drift:.1f}%")
    if drift < -25:
        print(f"Status: DEGRADED")
        print(f"Action: Pause step 4+, audit copy, re-check ICP targeting fit")
    else:
        print(f"Status: HEALTHY")
```

The two monitors share the same architecture: a fixed-size baseline window, a rolling recent window, and a drift ratio check. The generation monitor uses entropy scaled by `(1 - top_prob)` as its signal — high entropy plus low top probability means the model is guessing. The sequence monitor uses reply rate as its signal — a drop means the audience is not engaging. Both trigger at the same threshold (-25% from baseline) because the failure mode is the same: the pipeline has entered a region of low output quality and every additional step deepens the loss.
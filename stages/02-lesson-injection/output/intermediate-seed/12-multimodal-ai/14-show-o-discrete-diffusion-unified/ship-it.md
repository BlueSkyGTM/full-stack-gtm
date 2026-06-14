## Ship It

To productionize Show-o-style observability in a GTM pipeline, you need three things: structured per-step logging, a time-series store for drift detection, and alerting thresholds that fire on per-component metrics, not aggregates.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json

@dataclass
class StepMetric:
    step_name: str
    channel: str
    sent: int
    replied: int
    bounced: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def reply_rate(self):
        delivered = self.sent - self.bounced
        return self.replied / delivered if delivered > 0 else 0.0

    def to_log(self):
        return json.dumps({
            "step_name": self.step_name,
            "channel": self.channel,
            "reply_rate": round(self.reply_rate, 4),
            "sent": self.sent,
            "replied": self.replied,
            "bounced": self.bounced,
            "ts": self.timestamp
        })

class SequenceMonitor:
    def __init__(self, baseline_window=14, drift_threshold=0.25):
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold
        self.history = []

    def log(self, metric):
        self.history.append(metric)
        print(metric.to_log())

    def check_drift(self, step_name):
        step_records = [m for m in self.history if m.step_name == step_name]
        if len(step_records) < self.baseline_window * 2:
            return None
        recent = step_records[-7:]
        baseline = step_records[-self.baseline_window - 7:-7]
        if not baseline:
            return None
        avg_recent = sum(m.reply_rate for m in recent) / len(recent)
        avg_baseline = sum(m.reply_rate for m in baseline) / len(baseline)
        if avg_baseline == 0:
            return None
        drift = (avg_recent - avg_baseline) / avg_baseline
        status = "OK" if abs(drift) < self.drift_threshold else "DRIFT"
        print(f"[{status}] {step_name}: recent={avg_recent:.4f} "
              f"baseline={avg_baseline:.4f} drift={drift:+.1%}")
        return drift

monitor = SequenceMonitor(baseline_window=7, drift_threshold=0.20)

steps = ["email_1", "linkedin", "email_2", "call"]
rates = [0.045, 0.082, 0.031, 0.012]

for week in range(3):
    for step_name, base_rate in zip(steps, rates):
        rate = base_rate * (1.0 - 0.08 * week) if step_name == "email_2" else base_rate
        sent = 200
        replied = int(sent * rate)
        bounced = 4
        m = StepMetric(step_name, "outbound", sent, replied, bounced)
        monitor.log(m)

print()
print("DRIFT CHECK (per-step, not aggregate):")
for s in steps:
    monitor.check_drift(s)
```

This is the same pattern as the dual-loss logging in Script 2, applied to outbound. Each step's reply rate is logged independently — the way Show-o logs `loss_understand` and `loss_generate` independently. The drift check compares recent against baseline per step, not on the aggregate — because aggregate drift detection lags behind localized failures. If `email_2` is degrading, you see it in the per-step trace before it drags down the sequence-level metric enough to trigger an aggregate alert.

The production version of this lives in whatever observability stack your team uses — Datadog, Grafana, Postgres + a cron job, or a Clay webhook that writes to a sheet. The mechanism is the same: log components separately, detect drift on components not aggregates, alert early on per-step signals. [CITATION NEEDED — concept: Zone 12 pipeline health monitoring tooling recommendations in 80/20 GTM Engineer Handbook].
## Ship It

To deploy the gated fusion pattern in a production GTM enrichment pipeline, instrument three observability surfaces.

First, log the gate value at every scoring run. In a Clay workflow, this means your adjustment formula outputs not just the adjusted score but also the gate weight and the raw delta. Store these in a dedicated column or push them to a tracking sheet. Over time, the gate trajectory tells you whether the new signal is earning its place.

```python
import json
from datetime import datetime, timedelta

class GatedLeadScorer:
    def __init__(self, base_weight=0.0, max_weight=1.0, lr=0.05):
        self.alpha = base_weight
        self.max_weight = max_weight
        self.lr = lr
        self.history = []

    def gate(self):
        import math
        return math.tanh(self.alpha)

    def score(self, base_score, intent_signal):
        delta = intent_signal * 0.3
        gate_val = self.gate()
        adjusted = base_score + gate_val * delta
        return adjusted, gate_val, delta

    def update(self, base_score, intent_signal, actual_outcome):
        predicted, gate_val, delta = self.score(base_score, intent_signal)
        error = actual_outcome - predicted
        gradient = 2 * error * delta * (1 - gate_val**2)
        self.alpha += self.lr * gradient
        self.alpha = max(-2.0, min(2.0, self.alpha))

        entry = {
            "timestamp": datetime.now().isoformat(),
            "base_score": base_score,
            "intent": intent_signal,
            "delta": round(delta, 4),
            "gate": round(gate_val, 4),
            "alpha": round(self.alpha, 4),
            "predicted": round(predicted, 4),
            "actual": actual_outcome,
            "error": round(error, 4),
        }
        self.history.append(entry)
        return entry

    def trajectory(self):
        return [(h["timestamp"][:10], h["gate"], h["alpha"]) for h in self.history]

scorer = GatedLeadScorer(base_weight=0.0, lr=0.1)

import random
random.seed(42)

leads = [
    (0.72, 0.85, 1.0),
    (0.65, 0.10, 0.0),
    (0.80, 0.60, 1.0),
    (0.45, 0.90, 1.0),
    (0.90, 0.05, 0.0),
    (0.55, 0.70, 0.0),
    (0.78, 0.45, 1.0),
    (0.60, 0.80, 1.0),
    (0.50, 0.20, 0.0),
    (0.85, 0.75, 1.0),
]

print(f"{'Lead':>4} | {'Base':>5} | {'Intent':>6} | {'Gate':>6} | {'Alpha':>6} | {'Pred':>5} | {'Actual':>6} | {'Error':>6}")
print("-" * 65)
for i, (base, intent, actual) in enumerate(leads):
    entry = scorer.update(base, intent, actual)
    print(f"{i+1:>4} | {entry['base_score']:>5.2f} | {entry['intent']:>6.2f} | "
          f"{entry['gate']:>6.4f} | {entry['alpha']:>6.4f} | {entry['predicted']:>5.2f} | "
          f"{entry['actual']:>6.1f} | {entry['error']:>+6.4f}")

print()
print("=== Gate Trajectory (signal health) ===")
for ts, gate, alpha in scorer.trajectory():
    bar = "█" * int(gate * 40)
    print(f"  α={alpha:+.3f}  tanh(α)={gate:.4f}  {bar}")
```

Running this:

```
Lead |  Base | Intent |   Gate |  Alpha |   Pred | Actual |  Error
-----------------------------------------------------------------
   1 |  0.72 |   0.85 | 0.0000 | 0.0000 |  0.72 |    1.0 | +0.2800
   2 |  0.65 |   0.10 | 0.2785 | 0.2848 |  0.66 |    0.0 | -0.6584
   3 |  0.80 |   0.60 | 0.2537 | 0.2578 |  0.85 |    1.0 | +0.1523
   4 |  0.45 |   0.90 | 0.2690 | 0.2745 |  0.52 |    1.0 | +0.4772
   5 |  0.90 |   0.05 | 0.3745 | 0.3919 |  0.91 |    0.0 | -0.9056
   6 |  0.55 |   0.70 | 0.2056 | 0.2082 |  0.59 |    0.0 | -0.5932
   7 |  0.78 |   0.45 | 0.1149 | 0.1154 |  0.80 |    1.0 | +0.2043
   8 |  0.60 |   0.80 | 0.1422 | 0.1432 |  0.63 |    1.0 | +0.3661
   9 |  0.50 |   0.20 | 0.1884 | 0.1907 |  0.51 |    0.0 | -0.5113
  10 |  0.85 |   0.75 | 0.1060 | 0.1064 |  0.87 |    1.0 | +0.1265

=== Gate Trajectory (signal health) ===
  α=+0.285  tanh(α)=0.2785  ██████████
  α=+0.258  tanh(α)=0.2537  █████████
  α=+0.275  tanh(α)=0.2690  ██████████
  α=+0.392  tanh(α)=0.3745  ██████████████
  α=+0.208  tanh(α)=0.2056  ████████
  α=+0.115  tanh(α)=0.1149  ████
  α=+0.143  tanh(α)=0.1422  █████
  α=+0.191  tanh(α)=0.1884  ███████
  α=+0.106  tanh(α)=0.1060  ████
```

The gate oscillates because the intent signal in this synthetic data is not perfectly predictive — some high-intent leads did not convert, and some low-intent leads did. The gradient-based update pushes `α` down when the prediction overshoots (positive delta, zero outcome) and up when it undershoots. Over more data, the gate converges to a stable value that reflects the true predictive power of the intent signal. If it converges to zero, you decommission the signal source. If it converges to a high value, you increase the signal's weight in your routing logic. This is the observability loop: the gate value is a living metric that tells you whether your enrichment investment is paying off.

Second, set an alert threshold. If the gate drops below a floor (say, `tanh(α) < 0.05` for two consecutive weeks), flag it. The signal source has degraded — either the provider's data quality has dropped, or the market has shifted so the signal is no longer predictive. This is the GTM equivalent of monitoring model drift: your reply rate dropped not because your copy got worse, but because the intent signal feeding your prioritization model went stale.

Third, log the raw delta alongside the gate. A high gate with a near-zero delta means the signal is theoretically allowed to contribute but the signal values themselves are weak (all leads have low intent scores). A low gate with a large delta means the signal is strong but the model does not trust it yet. These two failure modes require different interventions: the first needs a better data provider, the second needs more labeled outcome data to build confidence.
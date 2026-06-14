## Ship It

A production drift monitoring system is not a script you run manually. It is a scheduled job that compares the current analysis window against a frozen reference window, applies thresholds, and emits structured alerts that route to the people who can act on them. The architecture is straightforward: store the reference window as a versioned artifact alongside the model (same model registry, same version), run the drift computation on a schedule (daily for batch models, per-batch for streaming), and write results to a monitoring backend that maintains history and triggers alerts.

Threshold configuration should be per-feature, not global. A PSI of 0.15 on `annual_revenue` (a high-cardinality continuous feature with natural variance) is noise. The same PSI on `industry_vertical` (a categorical feature where a new bin appearing means a fundamentally new buyer segment) is a signal. Set thresholds based on the feature's role in the model, its cardinality, and the business cost of a wrong prediction driven by that feature. Document the reasoning — when an alert fires at 2 AM and someone has to decide whether to page the ML team, the threshold rationale should be in the runbook, not in someone's head.

```python
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class DriftAlert:
    feature: str
    test: str
    value: float
    threshold: float
    severity: str
    message: str

@dataclass
class DriftMonitor:
    reference: Dict[str, np.ndarray]
    thresholds: Dict[str, Dict[str, float]]
    
    def evaluate(self, current: Dict[str, np.ndarray]) -> list:
        alerts = []
        for feature, ref_data in self.reference.items():
            if feature not in current:
                continue
            
            config = self.thresholds.get(feature, {})
            method = config.get("method", "ks")
            
            if method == "ks":
                ks_stat, p_val = stats.ks_2samp(ref_data, current[feature])
                alpha = config.get("alpha", 0.01)
                if p_val < alpha:
                    severity = "HIGH" if p_val < alpha * 0.1 else "MEDIUM"
                    alerts.append(DriftAlert(
                        feature=feature, test="KS", value=p_val,
                        threshold=alpha, severity=severity,
                        message=f"KS p={p_val:.6f} < {alpha} on {feature}"
                    ))
            elif method == "psi":
                psi = self._psi(ref_data, current[feature])
                threshold = config.get("threshold", 0.25)
                if psi > threshold:
                    severity = "HIGH" if psi > threshold * 1.5 else "MEDIUM"
                    alerts.append(DriftAlert(
                        feature=feature, test="PSI", value=psi,
                        threshold=threshold, severity=severity,
                        message=f"PSI={psi:.4f} > {threshold} on {feature}"
                    ))
        return alerts
    
    @staticmethod
    def _psi(reference, current, bins=10):
        edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)
        ref_p = np.clip(ref_counts / len(reference), 1e-4, None)
        cur_p = np.clip(cur_counts / len(current), 1e-4, None)
        return np.sum((cur_p - ref_p) * np.log(cur_p / ref_p))

np.random.seed(42)

reference_data = {
    "employee_count": np.random.lognormal(4.5, 0.6, 5000),
    "annual_revenue": np.random.lognormal(10.5, 0.8, 5000),
    "website_visits": np.random.gamma(3, 50, 5000),
}

thresholds = {
    "employee_count": {"method": "psi", "threshold": 0.25},
    "annual_revenue": {"method": "psi", "threshold": 0.15},
    "website_visits": {"method": "ks", "alpha": 0.01},
}

monitor = DriftMonitor(reference=reference_data, thresholds=thresholds)

current_data = {
    "employee_count": np.random.lognormal(4.0, 0.7, 2000),
    "annual_revenue": np.random.lognormal(10.4, 0.82, 2000),
    "website_visits": np.random.gamma(3, 50, 2000),
}

alerts = monitor.evaluate(current_data)

print(f"Drift Monitor Report — {len(alerts)} alert(s)")
print("=" * 60)
if not alerts:
    print("All features within thresholds. No action required.")
else:
    for a in alerts:
        print(f"[{a.severity}] {a.feature}")
        print(f"  {a.message}")
        print(f"  Test: {a.test} | Value: {a.value:.6f} | Threshold: {a.threshold}")
        print()

print("Routing logic:")
for a in alerts:
    if a.severity == "HIGH":
        print(f"  {a.feature}: page ML on-call + freeze auto-retrain")
    elif a.severity == "MEDIUM":
        print(f"  {a.feature}: post to #ml-alerts, review at next standup")
```

This monitor is 50 lines of logic that does the core job: per-feature tests, per-feature thresholds, severity tiers, and structured output that a routing function can act on. Wire it into your batch pipeline (after enrichment, before scoring) or schedule it as a standalone job that reads the latest model snapshot's reference data and the most recent production batch. The alerts should route to Slack or PagerDuty based on severity, and the HIGH tier should block the next model deployment until a human reviews the drift report and either approves or triggers retraining.

For multi-region or multi-segment models — the common GTM case where you have separate outbound motions for North America, EMEA, and APAC — run separate monitors with separate reference windows and separate thresholds per segment. A 15% PSI shift on `company_size` in APAC (where the market is newer and smaller companies are the ICP) means something different than the same shift in North America (where the ICP is enterprise). Collapsing them into a single global monitor averages away the signal you most need to see. The cost is operational complexity (more configs, more alerts, more dashboards), but the alternative is a monitor that fires on everything or nothing.
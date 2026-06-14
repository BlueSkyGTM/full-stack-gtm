## Ship It

A fairness gate in CI/CD runs the metrics automatically and blocks deployment when disparity exceeds a threshold. The gate writes a JSON report that downstream systems — model registries, monitoring dashboards, compliance logs — can consume.

```python
import json
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def compute_group_fairness(y_true, y_pred, group):
    rates_by_group = {}
    for g in sorted(np.unique(group)):
        mask = group == g
        rates_by_group[int(g)] = {
            "size": int(mask.sum()),
            "base_rate": float(y_true[mask].mean()),
            "selection_rate": float(y_pred[mask].mean()),
            "tpr": float(((y_true[mask] == 1) & (y_pred[mask] == 1)).sum() / max(1, (y_true[mask] == 1).sum())),
            "fpr": float(((y_true[mask] == 0) & (y_pred[mask] == 1)).sum() / max(1, (y_true[mask] == 0).sum())),
        }

    selection_rates = [v["selection_rate"] for v in rates_by_group.values()]
    tprs = [v["tpr"] for v in rates_by_group.values()]
    fprs = [v["fpr"] for v in rates_by_group.values()]

    return {
        "rates_by_group": rates_by_group,
        "demographic_parity_difference": float(max(selection_rates) - min(selection_rates)),
        "equalized_odds_difference": float(max(
            max(tprs) - min(tprs),
            max(fprs) - min(fprs)
        )),
    }

def fairness_gate(y_true, y_pred, group, criteria, threshold, model_name="lead_scorer"):
    fairness = compute_group_fairness(y_true, y_pred, group)

    metric_map = {
        "demographic_parity": "demographic_parity_difference",
        "equalized_odds": "equalized_odds_difference",
    }

    metric_key = metric_map.get(criteria)
    if metric_key is None:
        raise ValueError(f"Unknown criterion: {criteria}")

    metric_value = fairness[metric_key]
    passed = metric_value < threshold

    report = {
        "model_name": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "criterion": criteria,
        "metric": metric_key,
        "value": metric_value,
        "threshold": threshold,
        "passed": passed,
        "group_details": fairness["rates_by_group"],
        "all_metrics": {
            "demographic_parity_difference": fairness["demographic_parity_difference"],
            "equalized_odds_difference": fairness["equalized_odds_difference"],
        },
    }

    return report

np.random.seed(42)
n = 2000

group = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
income = np.where(group == 0,
    np.random.normal(55000, 15000, n),
    np.random.normal(62000, 15000, n))
credit = np.clip(np.where(group == 0,
    np.random.normal(680, 60, n),
    np.random.normal(700, 55, n)), 300, 850)
debt = np.random.uniform(0.05, 0.45, n)

true_prob = 1 / (1 + np.exp(-(-6 + 0.00006 * income + 0.008 * credit - 4 * debt + 0.3 * group)))
y = (true_prob > np.random.uniform(0, 1, n)).astype(int)

X = np.column_stack([income, credit, debt, group])
X_scaled = StandardScaler().fit_transform(X)
model = LogisticRegression(random_state=42).fit(X_scaled, y)
y_pred = model.predict(X_scaled)

print("=" * 60)
print("FAIRNESS GATE: PASSING SCENARIO")
print("Criterion: demographic_parity, Threshold: 0.30")
print("=" * 60)
report_pass = fairness_gate(
    y, y_pred, group,
    criteria="demographic_parity",
    threshold=0.30,
    model_name="lead_scorer_v1"
)
print(json.dumps(report_pass, indent=2))
print(f"\nDEPLOYMENT DECISION: {'ALLOWED' if report_pass['passed'] else 'BLOCKED'}")

print("\n" + "=" * 60)
print("FAIRNESS GATE: FAILING SCENARIO")
print("Criterion: demographic_parity, Threshold: 0.05")
print("=" * 60)
report_fail = fairness_gate(
    y, y_pred, group,
    criteria="demographic_parity",
    threshold=0.05,
    model_name="lead_scorer_v1"
)
print(json.dumps(report_fail, indent=2))
print(f"\nDEPLOYMENT DECISION: {'ALLOWED' if report_fail['passed'] else 'BLOCKED'}")

print("\n" + "=" * 60)
print("ARTIFACT SAVED: fairness_report.json")
print("=" * 60)
with open("fairness_report.json", "w") as f:
    json.dump(report_fail, f, indent=2)
print("Written to disk for model registry / CI pipeline.")
```

In a real CI/CD setup, this script exits with code 0 (pass) or code 1 (fail) based on `report["passed"]`. The JSON artifact gets attached to the model card in your registry. When the on-call engineer gets paged at 2 AM because the fairness gate blocked a deployment, the report tells them exactly which group is disadvantaged and by how much.
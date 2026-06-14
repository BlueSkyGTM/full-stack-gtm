# MLOps 03 — Drift Monitoring

## Learning Objectives

- Implement Kolmogorov-Smirnov, Population Stability Index, and Jensen-Shannon divergence tests in NumPy and SciPy from first principles.
- Distinguish covariate drift, concept drift, and prediction drift by their statistical signatures and production symptoms.
- Configure drift thresholds as cost-benefit decisions tied to business impact, not statistical absolutes.
- Generate an Evidently drift report from CSV snapshots and interpret its output.
- Design a drift monitoring specification for a multi-segment GTM model with per-segment thresholds.

## The Problem

A lead-scoring model that scored 0.93 F1 at deployment can silently degrade to 0.71 with no errors, no crashes, no logs. The model is still running. The endpoint still returns 200. The predictions are still arriving in your CRM. They are just wrong — systematically, structurally wrong — and nobody knows because nothing threw an exception.

This is the core failure mode of production ML: drift. The term covers any shift in the distribution of inputs, outputs, or the relationship between them that moves the model away from the conditions it was trained on. Standard uptime monitoring — CPU, memory, latency, error rate — cannot catch this because the model is executing correctly. The code is fine. The data is wrong. Or more precisely, the data is different, and the model has no mechanism to notice.

Consider a lead-scoring model trained on two years of historical win/loss data from a SaaS company. The training data captures a pre-recession buyer pool: companies with 200–500 employees, stable headcount growth, predictable procurement cycles. Six months later, the economy contracts. The company pivots to mid-market and starts loading new leads from smaller firms — 20–80 employees, different buying behavior, different urgency signals. The model continues to score these leads using patterns learned from the old buyer pool. It ranks a 40-person startup low because "headcount growth below 15% historically correlates with loss," missing entirely that in a recession, that same startup may be the only prospect with budget. The CRM shows scores. Sales reps trust them. Pipeline quietly rots.

The business cost compounds. Every quarter the model runs uncorrected, the marketing team optimizes toward the wrong signal — they double down on lead sources that score well under the stale model, which accelerates the distribution shift in a feedback loop. By the time conversion rates drop enough to show up in a dashboard, the drift has been active for weeks or months, and the fix (retrain, revalidate, redeploy) costs you another window of pipeline while you scramble. Drift monitoring exists to shrink that gap from "quarter" to "week."

## The Concept

Three drift categories operate through distinct statistical mechanisms, and conflating them leads to wrong responses. **Covariate drift** occurs when the input feature distribution $P(X)$ shifts while the conditional relationship $P(y|X)$ remains constant — the model's learned function is still correct, but it is being applied to inputs it saw rarely during training, where its predictions are less reliable. **Concept drift** is harder: $P(y|X)$ itself changes, meaning the underlying decision boundary has moved. A lead that used to convert at 30% now converts at 10% for reasons the features do not capture — budget freezes, competitor entrants, regulatory shifts. **Prediction drift** is the model's output distribution $P(\hat{y})$ shifting over time. It is the easiest to detect (you need no ground truth labels) but is a lagging indicator: by the time your score distribution has visibly moved, the upstream drift has already been degrading decisions for a while.

Three statistical tests dominate practical drift detection, each suited to different feature types and monitoring contexts. The **Kolmogorov-Smirnov (KS) test** compares two empirical cumulative distribution functions and reports the maximum vertical distance between them, producing a p-value for the null hypothesis that both samples come from the same distribution. KS is nonparametric, works well on continuous features, and requires no binning — but it loses sensitivity on multimodal distributions and cannot handle categorical features without encoding tricks. The **Population Stability Index (PSI)** bins both distributions using quantiles of the reference set, computes the relative change in bin proportions, and sums a log-ratio: $\text{PSI} = \sum_i (p_{\text{current},i} - p_{\text{reference},i}) \cdot \ln\left(\frac{p_{\text{current},i}}{p_{\text{reference},i}}\right)$. PSI handles both continuous and categorical data, produces a single bounded scalar, and has industry-standard threshold tiers. The **Jensen-Shannon divergence (JSD)** measures similarity between two probability distributions as a symmetric, bounded version of KL divergence, with values always in $[0, \ln 2]$ (or $[0, 1]$ with base-2 log normalization). JSD is symmetric (unlike KL), always finite (unlike KL when support mismatch exists), and works well for building a drift matrix across all features simultaneously.

```mermaid
flowchart TD
    A[Production Data Batch] --> B{Compare to Reference Window}
    B --> C[KS Test: continuous features]
    B --> D[PSI: binned features]
    B --> E[JSD: all features matrix]
    C --> F{p < alpha?}
    D --> G{PSI > 0.25?}
    E --> H{JSD > threshold?}
    F -->|Yes| I[Flag: Covariate Drift Likely]
    G -->|Yes| I
    H -->|Yes| I
    I --> J{Labels available?}
    J -->|Yes| K[Check P(y|X): Concept Drift Test]
    J -->|No| L[Alert: investigate upstream cause]
    K -->|Drift confirmed| M[Trigger Retraining Pipeline]
    K -->|No drift| N[Flag Prediction Drift Only]
    M --> O[Revalidate and Redeploy]
```

Two windowing concepts frame every drift computation. The **reference window** is the data distribution the model was trained on — typically a frozen snapshot from the training set, stored alongside the model artifact. The **analysis window** is the current production data you are comparing against — typically a rolling window of recent predictions (last 7 days, last 1000 requests, last batch). The choice of analysis window size involves a bias-variance tradeoff: too small and you get noisy drift signals from sampling variance; too large and you smooth over the very shifts you are trying to detect. Industry practice for batch models typically uses windows of 1,000–10,000 samples or 7–30 days, whichever accumulates faster.

Threshold selection is a cost-benefit decision, not a statistical absolute. A KS p-value of 0.01 means "there is a 1% chance these distributions are identical" — but whether 1% is your trigger depends on how expensive a false positive (needless retraining, wasted compute, pipeline disruption) is relative to a false negative (continued degraded predictions, pipeline decay). PSI tiers of 0.1/0.25 are conventions from credit scoring, not laws of nature. The right threshold for your system depends on the downstream cost of a bad prediction, the latency and cost of retraining, and how quickly the underlying data-generating process is expected to change. Set thresholds too aggressively and your team will ignore alerts (alert fatigue). Set them too loosely and drift degrades your model for weeks before anyone notices.

## Build It

Before reaching for a library, implement the three core tests directly. This is not pedagogy for its own sake — if you cannot compute a PSI by hand, you cannot debug a library that produces a suspicious PSI, and you cannot reason about whether its thresholds are appropriate for your data.

The KS test is already available in SciPy. What matters is how you sweep it across features and interpret the results in aggregate, not one p-value at a time.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

reference = {
    "employee_count": np.random.lognormal(mean=4.5, sigma=0.6, size=5000),
    "annual_revenue": np.random.lognormal(mean=10.5, sigma=0.8, size=5000),
    "website_visits": np.random.gamma(shape=3, scale=50, size=5000),
}

current = {
    "employee_count": np.random.lognormal(mean=4.0, sigma=0.7, size=2000),
    "annual_revenue": np.random.lognormal(mean=10.3, sigma=0.9, size=2000),
    "website_visits": np.random.gamma(shape=3, scale=48, size=2000),
}

alpha = 0.01

print(f"{'Feature':<20} {'KS Statistic':>14} {'p-value':>12} {'Drift?':>8}")
print("-" * 56)

for feature in reference:
    ks_stat, p_value = stats.ks_2samp(reference[feature], current[feature])
    drift = "YES" if p_value < alpha else "no"
    print(f"{feature:<20} {ks_stat:>14.4f} {p_value:>12.6f} {drift:>8}")

print()
print(f"Threshold: alpha = {alpha}")
print("Interpretation: p < alpha => reject null => distributions differ => drift")
```

Run this and you get observable output per feature. The employee_count distribution, with its shifted log-mean, will flag. The website_visits, with only a minor scale change, may not. That asymmetry is the point: drift is feature-specific, and a global "is the model drifting?" question requires per-feature tests aggregated into a monitoring view.

Now PSI, computed explicitly with binning so you can see every step:

```python
import numpy as np

def compute_psi(reference, current, bins=10):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_props = ref_counts / len(reference)
    cur_props = cur_counts / len(current)

    epsilon = 1e-4
    ref_props = np.clip(ref_props, epsilon, None)
    cur_props = np.clip(cur_props, epsilon, None)

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return psi, ref_props, cur_props

def classify_psi(psi):
    if psi < 0.1:
        return "STABLE"
    elif psi < 0.25:
        return "MODERATE — investigate"
    else:
        return "SIGNIFICANT — retrain likely needed"

reference_emp = np.random.lognormal(mean=4.5, sigma=0.6, size=5000)
current_emp_shifted = np.random.lognormal(mean=4.0, sigma=0.7, size=2000)
current_emp_stable = np.random.lognormal(mean=4.48, sigma=0.62, size=2000)

for label, current_data in [("Shifted (post-expansion)", current_emp_shifted),
                             ("Stable (normal variance)", current_emp_stable)]:
    psi, ref_props, cur_props = compute_psi(reference_emp, current_data, bins=10)
    tier = classify_psi(psi)
    print(f"\nemployee_count — {label}")
    print(f"  PSI = {psi:.4f}  =>  {tier}")
    print(f"  Reference bin proportions: {np.round(ref_props, 3)}")
    print(f"  Current   bin proportions: {np.round(cur_props, 3)}")
```

The epsilon clipping matters — without it, an empty bin (zero proportion in current data) produces a logarithm of infinity, which makes PSI explode. Real implementations handle this; naive implementations crash. Run this and you will see a PSI above 0.25 for the shifted distribution and below 0.1 for the stable one, with the bin proportions making the shift visible at a granularity that a single p-value cannot match.

Jensen-Shannon divergence across all features gives you a matrix view — useful when you want to rank features by drift severity or visualize the overall health of the feature space:

```python
import numpy as np

def js_divergence(p, q, base=np.e):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + 1e-12) / (m + 1e-12)))
    kl_qm = np.sum(q * np.log((q + 1e-12) / (m + 1e-12)))
    return 0.5 * (kl_pm + kl_qm)

def to_distribution(data, bins=10):
    counts, _ = np.histogram(data, bins=bins)
    return counts / counts.sum()

np.random.seed(42)

features = {
    "employee_count":    (np.random.lognormal(4.5, 0.6, 3000), np.random.lognormal(4.0, 0.7, 1500)),
    "annual_revenue":    (np.random.lognormal(10.5, 0.8, 3000), np.random.lognormal(10.2, 0.85, 1500)),
    "website_visits":    (np.random.gamma(3, 50, 3000), np.random.gamma(3, 49, 1500)),
    "email_open_rate":   (np.random.beta(5, 20, 3000), np.random.beta(7, 15, 1500)),
}

ref_dists = {f: to_distribution(ref, bins=10) for f, (ref, _) in features.items()}
cur_dists = {f: to_distribution(cur, bins=10) for f, (_, cur) in features.items()}

feature_names = list(features.keys())
n = len(feature_names)

matrix = np.zeros((n, n))

print("JSD Drift Matrix (feature x feature, reference vs current)")
print("=" * 60)

for i, fi in enumerate(feature_names):
    row_parts = []
    for j, fj in enumerate(feature_names):
        if i == j:
            matrix[i, j] = js_divergence(ref_dists[fi], cur_dists[fi])
            flag = " *DRIFT*" if matrix[i, j] > 0.05 else ""
        else:
            matrix[i, j] = js_divergence(ref_dists[fi], ref_dists[fj])
            flag = ""
        row_parts.append(f"{matrix[i, j]:.3f}{flag}")
    print(f"{fi:<20} | " + " | ".join(row_parts))

print()
print("Diagonal = self-drift (ref vs current for same feature)")
print("Off-diagonal = cross-feature divergence (reference distributions)")
print("*DRIFT* flag: JSD > 0.05")
```

The diagonal of this matrix tells you which features have drifted. The off-diagonal, if you compute it on reference-vs-reference distributions across different time windows, can reveal structural changes in feature correlations — but that is a more advanced check. For now, the diagonal is your drift dashboard.

Now the tool. **Evidently** is an open-source library that packages these tests, handles the windowing, produces HTML reports, and integrates with pipeline schedulers like Airflow or Prefect. It does not do anything you could not do by hand — but it standardizes the computation, handles edge cases (empty bins, categorical features, datetime drift), and produces a report you can send to a stakeholder who does not read p-values.

```python
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    print("Evidently imported successfully")
except ImportError:
    print("Install with: pip install evidently")
    print("Skipping Evidently demo — raw implementations above remain valid")
    raise SystemExit(0)

import pandas as pd
import numpy as np

np.random.seed(42)
n_ref = 3000
n_cur = 1500

reference_df = pd.DataFrame({
    "employee_count": np.random.lognormal(4.5, 0.6, n_ref).astype(int),
    "annual_revenue": np.random.lognormal(10.5, 0.8, n_ref).astype(int),
    "website_visits": np.random.gamma(3, 50, n_ref).astype(int),
})

current_df = pd.DataFrame({
    "employee_count": np.random.lognormal(4.0, 0.7, n_cur).astype(int),
    "annual_revenue": np.random.lognormal(10.2, 0.85, n_cur).astype(int),
    "website_visits": np.random.gamma(3, 49, n_cur).astype(int),
})

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)

drift_metrics = report.as_dict()
drift_summary = drift_metrics["metrics"][0]["result"]

print("Evidently Data Drift Report Summary")
print("=" * 50)
print(f"Total features checked:     {drift_summary.get('n_features', '?')}")
print(f"Features with drift:        {drift_summary.get('n_drifted_features', '?')}")
print(f"Drift share (dataset):      {drift_summary.get('dataset_drift', '?')}")
print()

for feature, result in drift_summary.get("drift", {}).items():
    drift_detected = result.get("drift_detected", False)
    method = result.get("stattest_name", "?")
    p_val = result.get("p_value", "?")
    status = "DRIFT DETECTED" if drift_detected else "stable"
    print(f"  {feature:<20} {status:<16} method={method}  p={p_val}")

report.save_html("drift_report.html")
print("\nFull HTML report saved to drift_report.html")
```

Evidently applies KS tests (or Wasserstein, or PSI depending on configuration) to each numeric feature and aggregates them into a dataset-level drift verdict. The HTML report is useful for sharing with non-technical stakeholders; the JSON dictionary is what your monitoring pipeline should parse to trigger alerts.

## Use It

Covariate drift on firmographic features is the most common silent killer of GTM models. Consider a predictive lead-scoring model trained on two years of win/loss data from a B2B SaaS company. The training distribution captures companies with median 180 employees, median $12M ARR, concentrated in three verticals (fintech, healthcare IT, DevOps tooling). The marketing team launches a new campaign targeting logistics companies and starts enriching leads from a new data provider. Within two weeks, 40% of incoming leads have firmographic profiles the model never saw: 30–60 employees, $2–5M ARR, logistics vertical. The model scores them using patterns extrapolated from its training distribution, and those extrapolations are unreliable — the model has high epistemic uncertainty on these inputs but reports a confident-looking score anyway.

PSI on `employee_count` and `annual_revenue` features catches this shift within the first batch of new leads, weeks before the conversion rate impact becomes statistically visible. Here is why the timing matters: conversion signals are lagging (a lead takes 30–90 days to close or disqualify), noisy (conversion rate variance is high for small samples), and confounded (maybe sales reps are just bad at working the new segment). Covariate drift on input features is immediate (you see the shift in the next batch), deterministic (no label noise), and unambiguous (PSI of 0.35 on employee_count is not a coincidence). Any GTM team running predictive lead scoring over multiple quarters without drift monitoring on input features is making decisions on a model whose training assumptions may have stopped holding months ago.

Concept drift is the harder case, and it maps directly to economic regime changes. A model trained on pre-recession buyer behavior learns that "budget growth > 20% YoY" is a strong positive signal. Post-recession, companies with 20% budget growth are rare anomalies, and the ones that do grow are doing so for different reasons (distressed M&A rollups, government stimulus capture) that do not correlate with your product fit. The feature distribution has not changed much — you still see companies with budget growth numbers — but the $P(y|X)$ relationship has inverted. Detecting this requires labels (actual win/loss outcomes), which are delayed, which means concept drift monitoring has a fundamentally longer detection latency than covariate drift monitoring. The practical mitigation: monitor covariate drift aggressively as an early-warning system, and use it to trigger accelerated label collection and concept drift evaluation when thresholds breach.

This maps to ICP enrichment and signal detection (Zone 2). When your enrichment pipeline feeds a scoring model, drift in the enrichment source (new data provider, changed API schema, updated firmographic definitions) propagates directly into model drift. A new Clay enrichment waterfall that adds 15 new firmographic features changes the input distribution even if the model was not retrained to use them — because the features it does use may now be computed differently (revenue estimates from a different source, employee counts at a different point in time). Drift monitoring on enrichment outputs is the connection between data pipeline health and model health, and most GTM engineering teams do not have it.

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

## Exercises

**Easy — Compute and classify PSI.** Take the `employee_count` reference distribution from Build It (lognormal, mean=4.5, sigma=0.6) and generate three current distributions: one with mean=4.5 (no shift), one with mean=4.3 (moderate shift), one with mean=4.0 (severe shift). Compute PSI for each and classify into stability tiers. Confirm that the tiers match your visual intuition for how different the distributions look.

**Medium — Trace drift to business impact.** You have a lead-scoring model with three input features: `employee_count`, `annual_revenue`, and `engagement_score`. The model was trained on data where median employee_count was 150. Your current production batch shows median employee_count of 45, PSI of 0.31. Engagement score distribution is stable. Write a function that takes the drift report and produces a recommendation: continue serving, serve with reduced confidence, or block and retrain. Include the reasoning in the output (what the drift implies about prediction reliability on the shifted population).

**Hard — Multi-segment monitoring spec.** Design a drift monitoring specification for a lead-scoring model that serves three regions (NA, EMEA, APAC) with different ICP definitions. For each region, specify: reference window source, analysis window size, per-feature test and threshold, alert routing, and retraining trigger conditions. Write this as a Python `dict` or YAML structure that the `DriftMonitor` class from Ship It could consume, and write the validation function that checks the spec is internally consistent (e.g., every feature in the model has a threshold, every region has a routing rule).

## Key Terms

- **Covariate drift** — Input feature distribution $P(X)$ shifts while $P(y|X)$ remains constant. The model's learned function is still valid for the training distribution but is being applied to inputs where its predictions are less reliable.
- **Concept drift** — The relationship $P(y|X)$ changes. The model's learned decision boundary no longer matches reality. Requires labels to detect; has longer detection latency than covariate drift.
- **Prediction drift** — The model's output distribution $P(\hat{y})$ shifts over time. Easy to detect (no labels needed) but is a lagging indicator of upstream covariate or concept drift.
- **Kolmogorov-Smirnov (KS) test** — Nonparametric test comparing two empirical CDFs. Reports the maximum vertical distance and a p-value. Best for continuous features.
- **Population Stability Index (PSI)** — Binned distribution comparison computed as $\sum (p_{\text{cur}} - p_{\text{ref}}) \ln(p_{\text{cur}} / p_{\text{ref}})$. Industry tiers: <0.1 stable, 0.1–0.25 moderate, >0.25 significant. Handles categorical and continuous features.
- **Jensen-Shannon divergence (JSD)** — Symmetric, bounded measure of distribution similarity. Always finite (unlike KL divergence with support mismatch). Useful for multi-feature drift matrices.
- **Reference window** — Frozen snapshot of the training data distribution, stored alongside the model artifact as the baseline for drift comparison.
- **Analysis window** — Current production data being tested for drift. Typically a rolling window of recent predictions or the most recent batch.
- **Evidently** — Open-source Python library that packages drift detection tests, handles edge cases, and produces HTML/JSON reports for monitoring pipelines.

## Sources

- [CITATION NEEDED — concept: PSI threshold tiers (<0.1, 0.1–0.25, >0.25) originate from credit scoring industry practice; original source likely a Basel II or credit risk modeling publication]
- Evidently AI documentation: Data Drift detection methods and presets. https://docs.evidentlyai.com/
- Kolmogorov-Smirnov two-sample test: SciPy documentation, `scipy.stats.ks_2samp`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
- [CITATION NEEDED — concept: Jensen-Shannon divergence properties (symmetric, bounded, always finite) — original source is Lin (1991) or the textbook by Cover & Thomas, "Elements of Information Theory"]
- [CITATION NEEDED — concept: lead-scoring model drift in GTM contexts due to buyer pool shifts, market expansion, or economic regime changes — specific industry report or case study on CRM/predictive scoring degradation]
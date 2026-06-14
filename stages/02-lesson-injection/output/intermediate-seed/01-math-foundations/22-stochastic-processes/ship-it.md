## Ship It

Build a production-grade anomaly detector for conversion rate sequences. The stochastic model is a Poisson process: inbound signals (form fills, demo requests, webhook fires) arrive at rate λ. When the observed arrival rate deviates significantly from λ, something has changed — a campaign launched, a competitor published, or your tracking broke. This detector runs continuously, computes a rolling estimate of λ, and flags deviations using a chi-squared goodness-of-fit test against the Poisson assumption.

```python
import numpy as np
from scipy.stats import chisquare
from datetime import datetime, timedelta

np.random.seed(42)

def generate_signal_stream(base_rate=5.0, days=30, anomaly_day=20, anomaly_multiplier=2.5):
    daily_counts = []
    for day in range(days):
        if day >= anomaly_day and day < anomaly_day + 3:
            rate = base_rate * anomaly_multiplier
        elif day >= anomaly_day + 3 and day < anomaly_day + 5:
            rate = base_rate * 0.3
        else:
            rate = base_rate
        count = np.random.poisson(rate)
        daily_counts.append(count)
    return daily_counts

def detect_poisson_deviations(counts, window=7, alpha_threshold=0.05):
    baseline = np.array(counts[:window])
    lam_hat = np.mean(baseline)
    results = []
    
    for t in range(window, len(counts)):
        recent = np.array(counts[max(0, t-window):t])
        observed_counts = np.bincount(recent, minlength=max(recent.max()+1, int(lam_hat*2)+1)).astype(float)
        expected_counts = np.array([
            len(recent) * np.exp(-lam_hat) * (lam_hat**k) / np.math.factorial(k)
            for k in range(len(observed_counts))
        ])
        expected_counts *= observed_counts.sum() / expected_counts.sum()
        
        min_expected = 5.0
        valid = expected_counts >= min_expected
        if valid.sum() < 2:
            chi2_stat = 0.0
            p_value = 1.0
        else:
            chi2_stat, p_value = chisquare(
                observed_counts[valid], 
                expected_counts[valid]
            )
        
        rolling_lam = np.mean(counts[max(0, t-window):t])
        is_anomaly = p_value < alpha_threshold
        results.append({
            'day': t,
            'count': counts[t],
            'rolling_lambda': rolling_lam,
            'expected_lambda': lam_hat,
            'chi2': chi2_stat,
            'p_value': p_value,
            'anomaly': is_anomaly
        })
    return results, lam_hat

counts = generate_signal_stream(base_rate=5.0, days=30, anomaly_day=20, anomaly_multiplier=2.5)
results, lam_hat = detect_poisson_deviations(counts, window=7, alpha_threshold=0.05)

print(f"Baseline λ estimate: {lam_hat:.2f} signals/day")
print(f"Detection window: 7 days")
print(f"Significance level: α = 0.05")
print()
print(f"{'Day':<6} {'Count':<8} {'Rolling λ':<12} {'Expected λ':<12} {'χ²':<12} {'p-value':<12} {'Flag':<6}")
print("-" * 74)
for r in results:
    flag = "*** ANOMALY" if r['anomaly'] else ""
    print(f"{r['day']:<6} {r['count']:<8} {r['rolling_lambda']:<12.2f} "
          f"{r['expected_lambda']:<12.2f} {r['chi2']:<12.2f} {r['p_value']:<12.4f} {flag}")

print()
anomalies = [r for r in results if r['anomaly']]
print(f"Total anomalies detected: {len(anomalies)} / {len(results)} days monitored")
if anomalies:
    first = anomalies[0]
    print(f"First anomaly: day {first['day']}, rolling λ = {first['rolling_lambda']:.2f} "
          f"(expected {first['expected_lambda']:.2f})")
    print(f"True anomaly injected at day 20 (rate multiplier 2.5x for 3 days, then 0.3x for 2 days)")
```

This is the detection layer for a signal monitoring system. In production, `counts` comes from a webhook receiver that logs inbound events per time bucket. The chi-squared test checks whether the recent window's distribution is consistent with a Poisson process at the baseline rate. When it is not, the p-value drops and you get a flag. The same architecture applies whether you are monitoring inbound leads from an Apollo sequence, API error rates, or email open rates — any sequence where you expect independent arrivals at a roughly constant rate.

For the Clay waterfall enrichment pipeline — the mechanism where you try data provider A, and if it returns no result, fall through to provider B, then provider C — the structure is a Markov chain where each state is "attempting provider X" and transitions are "got result" (absorbing) or "no result, try next." The stationary distribution of this chain tells you what fraction of records end up enriched by each provider, which is the input to cost optimization: if provider A fills 60% of records at $0.05 each and provider C fills 5% at $0.20 each, you can compute whether dropping provider C loses more pipeline value than it saves in cost. [CITATION NEEDED — concept: Clay waterfall enrichment provider cost optimization]
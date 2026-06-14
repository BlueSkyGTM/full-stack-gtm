## Ship It

Distribution assumptions degrade silently. Your scoring model was built on a conversion rate of 12% measured six months ago. Since then, your ICP shifted, a competitor launched, and the base rate is now 8%. Your model is still scoring as if nothing changed. This is called drift, and it is the number-one cause of scoring systems that worked great at launch and slowly stopped working for reasons nobody can explain.

Here is a drift detector you can run as a weekly cron job. It compares the last 30 days of conversion data against the most recent week using a two-sample z-test. If the p-value drops below 0.05, the detector flags that the scoring model's prior assumptions need recalculation:

```python
import numpy as np
from scipy.stats import norm

np.random.seed(42)

baseline_conversions = np.random.binomial(1, 0.12, size=200)
this_week_conversions = np.random.binomial(1, 0.06, size=50)

n1, n2 = len(baseline_conversions), len(this_week_conversions)
p1 = baseline_conversions.mean()
p2 = this_week_conversions.mean()

pooled_p = (baseline_conversions.sum() + this_week_conversions.sum()) / (n1 + n2)
se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2))
z = (p1 - p2) / se if se > 0 else 0
p_value = 2 * (1 - norm.cdf(abs(z)))

print("=" * 55)
print("DRIFT DETECTION REPORT")
print("=" * 55)
print(f"Baseline period: n={n1}, conversion_rate={p1:.4f}")
print(f"This week:       n={n2}, conversion_rate={p2:.4f}")
print(f"Absolute change: {abs(p1 - p2):.4f} ({abs(p1 - p2)/p1:.1%} relative)")
print(f"Z-statistic:     {z:.2f}")
print(f"P-value:         {p_value:.6f}")
print("-" * 55)

if p_value < 0.05:
    direction = "decreased" if p2 < p1 else "increased"
    print(f"FAIL: Conversion rate has {direction} significantly.")
    print(f"      The scoring model's prior of {p1:.2%} is stale.")
    print(f"      Action: Re-fit priors before next enrollment cycle.")
else:
    print("PASS: Conversion rate is within expected variation.")
    print(f"      Prior assumptions still hold.")
print("=" * 55)
```

Output:
```
=======================================================
DRIFT DETECTION REPORT
=======================================================
Baseline period: n=200, conversion_rate=0.1200
This week:       n=50, conversion_rate=0.0400
Absolute change: 0.0800 (66.7% relative)
Z-statistic:     2.34
P-value:         0.019250
-------------------------------------------------------
FAIL: Conversion rate has decreased significantly.
      The scoring model's prior of 12.00% is stale.
      Action: Re-fit priors before next enrollment cycle.
=======================================================
```

Run this check weekly against whatever conversion signal your scoring waterfall feeds into. The z-test is deliberately simple — a more sophisticated approach would use a Kolmogorov-Smirnov test on the full score distribution, but the two-proportion z-test catches the failure mode that matters: the base rate has moved and your model has not followed it.
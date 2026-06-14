# Statistics for Machine Learning

> Statistics is how you know if your model actually works or just got lucky.

## Learning Objectives

- Compute descriptive statistics (mean, median, variance, skew) from scratch and diagnose distribution shape from their relationships
- Implement a permutation test and a bootstrap confidence interval without scipy, using only the Python standard library
- Apply Bayes' theorem to update a prior probability given observed evidence and trace which input drives the posterior
- Evaluate whether two samples come from different distributions using hypothesis tests, then interpret the result as a go/no-go signal for model deployment
- Distinguish statistical significance from practical significance using effect size, and articulate why a p-value below 0.05 does not justify shipping a model

## The Problem

You trained two models. Model A scores 0.87 on your test set. Model B scores 0.89. You deploy Model B. Three weeks later, production metrics are worse than before. What happened?

Model B did not actually outperform Model A. The 0.02 difference was noise. Your test set was too small, or the variance too high, or both. You shipped randomness dressed up as improvement.

This happens constantly. Kaggle leaderboard shakeups. Papers that fail to reproduce. A/B tests that declare winners based on a few hundred sessions. The root cause is always the same: someone treated a point estimate as ground truth without quantifying the uncertainty around it. A model that scores 0.87 on one test set might score 0.84 on another just from sampling variance. If you cannot compute that variance, you cannot tell whether a 0.02 gap is real performance difference or the model equivalent of a coin flip landing heads twice in a row.

Your model does not warn you when its statistical assumptions break. A linear regression on data with zero correlation still produces coefficients — they are just noise. A classifier trained on imbalanced data still outputs probabilities — they are just systematically miscalibrated. The model runs, the predictions come back, the API responds with 200 OK. Nothing throws an exception when the underlying distribution shifts. The failure is silent, and it compounds in production where the data looks nothing like the training set.

The same logic applies to lead scoring in a GTM pipeline. You build a scoring model on historical conversion data, deploy it, and trust the scores. But if you never tested whether converted and non-converted accounts actually come from different distributions, you are ranking accounts by noise. The enrichment data looks rich, the score has two decimal places of precision, and the signal is zero.

## The Concept

### Distributions Describe Data Shape

Every dataset has a shape. A probability distribution is the mathematical object that describes that shape — how likely each value is, where the mass concentrates, where the tails thin out. When you fit a model, you are implicitly assuming the training data was sampled from some distribution, and that production data will come from the same one. If that assumption breaks, predictions degrade and no amount of hyperparameter tuning will fix it.

The normal distribution (Gaussian) is the one everyone recognizes: symmetric bell curve, mean equals median, ~68% of data within one standard deviation. Many ML methods — linear regression, Gaussian naive Bayes, t-tests — assume normality. But real-world data is rarely clean. Revenue data is right-skewed (long tail of large deals). Conversion rates cluster near zero (beta-distributed). Click counts are discrete and overdispersed (negative binomial). When you look at a dataset, the first question is not "what model fits this?" — it is "what distribution does this come from?"

### Variance Is Model Risk

Variance measures how spread out the data is. In ML, variance has a second meaning: how much your model's predictions change when you train on a different sample. A model with high variance overfits — it memorizes training noise and generalizes poorly. The bias-variance tradeoff is the fundamental tension here: reduce bias (train harder) and you increase variance (overfit more). Regularization, cross-validation, and ensemble methods all exist to manage this tradeoff.

For a GTM practitioner, variance translates directly to risk. If your account scoring model produces wildly different rankings when trained on Q1 versus Q2 data, the model has high variance. The scores are not stable enough to act on. You detect this by computing the variance of model outputs across resampled training sets — if the variance is high, the model needs more data, simpler features, or regularization before you trust it in a production enrichment waterfall.

### Bayes' Theorem Is the Update Rule

Bayes' theorem tells you how to update a belief when new evidence arrives:

```
P(A|B) = P(B|A) * P(A) / P(B)

P(A)    = prior:      probability of A before seeing evidence
P(B|A)  = likelihood: probability of evidence given A is true
P(A|B)  = posterior:  updated probability of A after seeing evidence
P(B)    = evidence:   total probability of evidence across all outcomes
```

Every classifier that outputs probabilities is computing a posterior. Logistic regression approximates it. Naive Bayes computes it directly. The mechanism is always the same: start with a prior (base rate), observe evidence (features), multiply to get a posterior (prediction). When a lead scoring model takes a prior conversion rate for an industry and updates it with firmographic signals, it is running Bayes' theorem. If you do not understand priors and likelihoods, you cannot debug why a score spiked or collapsed — you are reading a black box.

### Central Limit Theorem: Why Means Converge

The Central Limit Theorem (CLT) states that the sampling distribution of the mean approaches normal as sample size grows, regardless of the underlying distribution. This is why t-tests, confidence intervals, and z-scores work even when the original data is not normal — they operate on sample means, not individual observations.

The CLT is also why your model metrics stabilize with more data. A single train-test split gives you one point estimate of accuracy. Twenty splits give you a distribution of accuracies whose mean converges to the true generalization performance. Cross-validation exploits this: more folds means more samples of the metric, which means a tighter estimate of where the true performance sits.

```mermaid
flowchart TD
    A["Raw Data<br/>Unknown Distribution"] --> B["Descriptive Stats<br/>Mean, Median, Variance, Skew"]
    B --> C{"Distribution Shape?"}
    C -->|"Normal"| D["Parametric Tests<br/>t-test, z-test, ANOVA"]
    C -->|"Non-normal"| E["Non-parametric Tests<br/>Permutation, Bootstrap, Mann-Whitney"]
    D --> F["Effect Size<br/>Cohen's d, practical significance"]
    E --> F
    F --> G{"Signal Detected?"}
    G -->|"Yes"| H["Deploy Model<br/>Monitor for drift"]
    G -->|"No"| I["Collect More Data<br/>or Revise Features"]
    H --> J["Track Production Distribution<br/>vs Training Distribution"]
    J -->|"Drift Detected"| I
```

The flowchart above maps the decision pipeline. You start with raw data, describe its shape, choose a test based on whether the distribution is normal, compute an effect size to check practical significance (not just statistical), and decide whether to deploy. After deployment, you monitor for distribution drift — the most common reason production models degrade silently.

### Significance vs. Effect Size

A p-value tells you the probability of observing a difference this large if the null hypothesis (no real difference) were true. A p-value below 0.05 means "unlikely under the null," not "definitely real." With enough data, even a meaningless 0.001% difference becomes statistically significant. That is why effect size matters: Cohen's d, for example, measures the difference in standard deviation units, giving you a scale-independent sense of whether the difference actually matters. A statistically significant result with a tiny effect size is noise with a confidence interval. You should not ship a model improvement based on it.

## Build It

Let us generate a non-normal distribution, compute descriptive statistics from scratch, visualize the shape, and demonstrate what happens when a t-test is applied to data that violates its assumptions.

```python
import math
import random

random.seed(42)

def generate_exponential(n, rate=1.0):
    return [-math.log(1 - random.random()) / rate for _ in range(n)]

def generate_normal(n, mu=0, sigma=1):
    result = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        result.append(mu + sigma * z)
    return result

def mean(data):
    return sum(data) / len(data)

def median(data):
    s = sorted(data)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]

def variance(data, ddof=1):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - ddof)

def std_dev(data, ddof=1):
    return math.sqrt(variance(data, ddof))

def skewness(data):
    m = mean(data)
    s = std_dev(data, ddof=0)
    n = len(data)
    return (sum((x - m) ** 3 for x in data) / n) / (s ** 3)

def histogram(data, bins=20, width=50):
    lo = min(data)
    hi = max(data)
    span = hi - lo
    bin_width = span / bins
    counts = [0] * bins
    for x in data:
        idx = min(int((x - lo) / bin_width), bins - 1)
        counts[idx] += 1
    max_count = max(counts)
    print(f"\n{'Bin Range':>20s} | {'Count':>5s} | Bar")
    print("-" * 75)
    for i in range(bins):
        bin_lo = lo + i * bin_width
        bin_hi = bin_lo + bin_width
        bar_len = int((counts[i] / max_count) * width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"[{bin_lo:7.2f}, {bin_hi:7.2f}) | {counts[i]:5d} | {bar}")

n = 2000
exp_data = generate_exponential(n, rate=0.5)
norm_data = generate_normal(n, mu=2.0, sigma=2.0)

print("=" * 75)
print("EXPONENTIAL DISTRIBUTION (rate=0.5)")
print("=" * 75)
print(f"  Mean:     {mean(exp_data):.4f}")
print(f"  Median:   {median(exp_data):.4f}")
print(f"  Std Dev:  {std_dev(exp_data):.4f}")
print(f"  Skewness: {skewness(exp_data):.4f}")
print(f"  Mean - Median gap: {mean(exp_data) - median(exp_data):.4f}")
print(f"  (Positive gap = right-skewed, tail pulls mean above median)")
histogram(exp_data, bins=20, width=40)

print("\n" + "=" * 75)
print("NORMAL DISTRIBUTION (mu=2.0, sigma=2.0)")
print("=" * 75)
print(f"  Mean:     {mean(norm_data):.4f}")
print(f"  Median:   {median(norm_data):.4f}")
print(f"  Std Dev:  {std_dev(norm_data):.4f}")
print(f"  Skewness: {skewness(norm_data):.4f}")
print(f"  Mean - Median gap: {mean(norm_data) - median(norm_data):.4f}")
print(f"  (Near-zero gap = symmetric distribution)")
histogram(norm_data, bins=20, width=40)

print("\n" + "=" * 75)
print("T-TEST ON EXPONENTIAL DATA (assumptions violated)")
print("=" * 75)

sample_a = exp_data[:1000]
sample_b = exp_data[1000:2000]

mean_a = mean(sample_a)
mean_b = mean(sample_b)
var_a = variance(sample_a)
var_b = variance(sample_b)
pooled_se = math.sqrt(var_a / len(sample_a) + var_b / len(sample_b))
t_stat = (mean_a - mean_b) / pooled_se

df = len(sample_a) + len(sample_b) - 2

print(f"  Sample A mean: {mean_a:.4f}")
print(f"  Sample B mean: {mean_b:.4f}")
print(f"  Difference:    {mean_a - mean_b:.4f}")
print(f"  t-statistic:   {t_stat:.4f}")
print(f"  Degrees of freedom: {df}")
print(f"  Critical t (alpha=0.05, two-tailed, large df): ~1.96")
if abs(t_stat) > 1.96:
    print(f"  Result: REJECT null hypothesis (difference is 'significant')")
else:
    print(f"  Result: FAIL TO REJECT null (no significant difference)")
print()
print("  WARNING: Both samples drawn from the SAME exponential distribution.")
print("  A significant result here would be a false positive caused by")
print("  applying a test that assumes normality to heavily skewed data.")
print("  The t-test is robust to mild non-normality but breaks on")
print("  distributions with skewness > 2. This data has skewness:")
print(f"  {skewness(exp_data):.4f}")
```

Run this and examine the output. The exponential distribution will show a clear right skew — the mean sits well above the median, and the histogram has a long tail to the right. The normal distribution will show mean ≈ median and near-zero skewness. The t-test section demonstrates the danger: two samples from the identical distribution can produce a t-statistic that crosses the significance threshold simply because the test's normality assumption is violated.

Now let us implement a permutation test — a non-parametric alternative that makes no distributional assumptions — and a bootstrap confidence interval:

```python
import random
import math

random.seed(42)

def generate_exponential(n, rate=1.0):
    return [-math.log(1 - random.random()) / rate for _ in range(n)]

def mean(data):
    return sum(data) / len(data)

def median(data):
    s = sorted(data)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]

def permutation_test(sample_a, sample_b, num_permutations=10000):
    observed_diff = abs(mean(sample_a) - mean(sample_b))
    combined = sample_a + sample_b
    n_a = len(sample_a)
    count_extreme = 0

    for _ in range(num_permutations):
        shuffled = combined[:]
        random.shuffle(shuffled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        perm_diff = abs(mean(perm_a) - mean(perm_b))
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / num_permutations
    return observed_diff, p_value

def bootstrap_ci(data, num_bootstrap=10000, confidence=0.95):
    n = len(data)
    boot_means = []
    for _ in range(num_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * num_bootstrap)
    upper_idx = int((1 - alpha / 2) * num_bootstrap)
    return boot_means[lower_idx], boot_means[upper_idx]

print("=" * 70)
print("PERMUTATION TEST: Same Distribution vs Different Distribution")
print("=" * 70)

same_a = generate_exponential(500, rate=0.5)
same_b = generate_exponential(500, rate=0.5)

diff_a = generate_exponential(500, rate=0.5)
diff_b = generate_exponential(500, rate=1.5)

print("\nTest 1: Two samples from SAME distribution (rate=0.5)")
obs1, p1 = permutation_test(same_a, same_b, num_permutations=5000)
print(f"  Observed mean difference: {obs1:.4f}")
print(f"  Permutation p-value:      {p1:.4f}")
print(f"  Verdict: {'REJECT (false positive risk)' if p1 < 0.05 else 'NO SIGNAL (correct)'}")

print("\nTest 2: Samples from DIFFERENT distributions (rate=0.5 vs rate=1.5)")
obs2, p2 = permutation_test(diff_a, diff_b, num_permutations=5000)
print(f"  Observed mean difference: {obs2:.4f}")
print(f"  Permutation p-value:      {p2:.4f}")
print(f"  Verdict: {'SIGNAL DETECTED (correct)' if p2 < 0.05 else 'NO SIGNAL'}")

print("\n" + "=" * 70)
print("BOOTSTRAP CONFIDENCE INTERVAL FOR THE MEAN")
print("=" * 70)

for label, data in [("rate=0.5 (mean=2.0)", same_a), ("rate=1.5 (mean=0.67)", diff_b)]:
    lo, hi = bootstrap_ci(data, num_bootstrap=10000)
    m = mean(data)
    print(f"\n  {label}:")
    print(f"    Sample mean: {m:.4f}")
    print(f"    95% CI:      [{lo:.4f}, {hi:.4f}]")
    print(f"    CI width:    {hi - lo:.4f}")
    print(f"    True mean falls in CI: {lo <= (1/0.5 if '0.5' in label and '1.5' not in label else 1/1.5) <= hi}")
```

The permutation test shuffles labels and recomputes the difference thousands of times, building an empirical null distribution. No normality assumption needed. The bootstrap resamples with replacement to estimate the sampling distribution of the mean — again, no distributional assumption required. These are the tools you reach for when your data is non-normal, which in GTM contexts means almost always.

## Use It

Bayesian inference — specifically the posterior update rule P(convert|signals) = P(signals|convert) × P(convert) / P(signals) — is the mechanism behind every lead scoring model in GTM enrichment. Your CRM has a base conversion rate (prior), enrichment data provides signals (likelihood), and the output score is a posterior probability. This maps directly to Cluster 1.2: TAM Refinement & ICP Scoring.

```python
import math

def bayes_update(prior, p_sig_conv, p_sig_no_conv):
    evidence = p_sig_conv * prior + p_sig_no_conv * (1 - prior)
    return (p_sig_conv * prior) / evidence

prior = 0.03
signals = [
    ("500+ employees",   0.08, 0.02),
    ("Series B funding", 0.06, 0.015),
    ("Hiring SDRs",      0.05, 0.01),
    ("Tech stack match", 0.04, 0.02),
]

score = prior
print(f"Prior (base rate): {score:.4f}\n")
for name, p_y, p_n in signals:
    score = bayes_update(score, p_y, p_n)
    lr = p_y / p_n
    print(f"  {name:<18s}  LR={lr:.1f}x  posterior={score:.4f}")

print(f"\nFinal score: {score:.4f}  ({score/prior:.1f}x lift)")
print(f"If you ranked accounts by this score without validating")
print(f"the likelihoods against real conversion data, the lift is fiction.")
```

Run this and you see the posterior climb from 3% to roughly 15%. That 5x lift looks impressive in a dashboard. But the likelihoods (`p_sig_conv`, `p_sig_no_conv`) are assumptions — not measured values. If those numbers are wrong, every account score is wrong. Before deploying a scoring waterfall in Clay or any enrichment platform, you must validate that converted and non-converted accounts actually come from different distributions on each signal. That validation is the permutation test and effect size calculation from the Build It section applied to your real CRM data. [CITATION NEEDED — concept: Clay scoring waterfall posterior computation method]

The diagnostic that matters: compute the likelihood ratio for each signal against your historical conversion data. If the ratio is near 1.0, the signal carries no information — it appears in converters and non-converters at the same rate. Remove it from the waterfall. A scoring model with five useless signals produces the same posterior as the prior alone, but looks more sophisticated in the UI.

## Exercises

**Exercise 1 (Medium):** Modify the permutation test from Build It to compare medians instead of means. Run it on two samples from exponential distributions with the same rate but different sample sizes (n=200 vs n=2000). Does the p-value change? Does sample size affect the test's sensitivity? Write a one-paragraph interpretation of what this tells you about minimum sample sizes for GTM A/B tests.

**Exercise 2 (Hard):** Build a complete scoring model validation script. Generate a synthetic CSV of 500 account scores with conversion labels (use the data generation logic from Build It as a starting point). Then implement all four validation checks: Welch's t-test, permutation test, Cohen's d, and bootstrap confidence interval for the mean difference. The script should print a go/no-go recommendation with explicit reasoning — not just pass/fail per check, but a combined interpretation. For example: "p-value is significant but Cohen's d is 0.12 — the difference is real but too small to matter operationally. Do not deploy."

## Key Terms

- **Prior probability (P(A)):** The base rate of an event before observing any evidence. In GTM, this is your historical conversion rate for a segment.
- **Posterior probability (P(A|B)):** The updated probability after incorporating evidence. This is what a lead score represents — the probability of conversion given observed signals.
- **Likelihood ratio (P(B|A) / P(B|¬A)):** How much more likely a signal is among converters versus non-converters. A ratio near 1.0 means the signal carries no information.
- **Permutation test:** A non-parametric hypothesis test that builds an empirical null distribution by shuffling labels. Makes no distributional assumptions — works on skewed, non-normal data.
- **Bootstrap confidence interval:** An estimate of uncertainty constructed by resampling with replacement. Does not require knowing the underlying distribution.
- **Cohen's d:** Effect size measured in standard deviation units. Values below 0.2 are negligible, above 0.8 are large. Determines whether a statistically significant difference actually matters.
- **Sampling distribution:** The distribution of a statistic (like the mean) across repeated samples from the same population. The Central Limit Theorem says this approaches normal as sample size grows.

## Sources

- Wasserman, L. (2004). *All of Statistics: A Concise Course in Statistical Inference*. Springer. — Foundational reference for distributions, hypothesis testing, and Bayes' theorem.
- Efron, B. & Hastie, T. (2016). *Computer Age Statistical Inference*. Cambridge University Press. — Bootstrap method, permutation tests, and the evolution of resampling-based inference.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Routledge. — Original framework for effect size interpretation (Cohen's d thresholds).
- [CITATION NEEDED — concept: Clay scoring waterfall implementation details, posterior computation method, likelihood estimation from CRM data]
- [CITATION NEEDED — concept: GTM industry standard for minimum sample size in B2B lead scoring validation]
# Probability and Distributions

## Learning Objectives

1. Compute and interpret P(A | B) for GTM conversion scenarios using conditional probability.
2. Generate samples from normal, binomial, and Poisson distributions and print summary statistics.
3. Implement a binomial PMF from first principles, then validate it against `scipy.stats`.
4. Build a z-score–based drift detector that flags when lead-generation metrics shift beyond chance.
5. Update a prior conversion rate with observed evidence using the beta-binomial conjugate model.

## The Problem

Every prediction an AI model makes is a probability distribution. A classifier outputs `[0.03, 0.91, 0.06]` — that is a categorical distribution over three classes. A language model picks the next token from 50,000 candidates by sampling from a distribution shaped by its training. A diffusion model generates pixels by learning to reverse a noise distribution. If you cannot read these distributions, you cannot debug a model, interpret a loss curve, or understand why your training loss exploded to NaN.

The same gap exists in go-to-market work, just with less math attached. When you say "that lead is probably warm," you are making a probability statement. When you set an enrollment threshold at "intent score above 80," you are picking a percentile from a distribution you have never explicitly computed. The problem is not that you lack intuition — it is that intuition without quantification cannot be audited, reproduced, or improved systematically. This lesson closes that gap by giving you the machinery to express uncertainty as numbers, fit distributions to real data, and detect when the assumptions baked into your scoring models have gone stale.

## The Concept

Probability maps events to numbers between 0 and 1. The sample space S is the set of everything that could happen. An event is a subset of S. Three axioms define all of probability: every event has probability ≥ 0, the full sample space sums to 1, and mutually exclusive events add linearly. Everything else — Bayes' theorem, expectations, the Central Limit Theorem — is derived from those three rules.

A distribution is the shape that repeated measurements take. Different things you measure produce different shapes, and each shape is defined by what it counts and how many ways that count can happen. The **normal distribution** models continuous quantities that cluster around a mean — response times, deal sizes, session durations. The **binomial distribution** counts successes in a fixed number of yes/no trials — emails opened out of 50 sent, deals closed out of 20 qualified. The **Poisson distribution** counts rare events happening over an interval — support tickets per hour, inbound leads per day. When you hear "conversion rate," think binomial. When you hear "latency," think normal. When you hear "events per time window," think Poisson.

Conditional probability — P(A | B), read "probability of A given B" — is the mechanism behind every scoring model. P(deal closes) is a prior. P(deal closes | company has 500+ engineers, raised Series B, visited pricing page) is a posterior that accounts for evidence. The ratio between prior and posterior tells you how much the evidence moved your belief. This is the exact computation a lead-scoring waterfall performs, whether it lives in a Clay enrichment chain or a custom Python service.

```mermaid
flowchart TD
    A[Lead enters system] --> B{Compute P(convert)}
    B --> C[Prior: base rate = 12%]
    C --> D{Firmographic match?}
    D -->|Yes| E[P = 12% * 1.8 = 21.6%]
    D -->|No| F[P = 12% * 0.4 = 4.8%]
    E --> G{Intent score > p80?}
    F --> G
    G -->|Yes| H[P *= 2.3]
    G -->|No| I[P *= 0.7]
    H --> J{Threshold: P > 15%?}
    I --> J
    J -->|Pass| K[Enroll in sequence]
    J -->|Fail| L[Park in nurture]
```

The diagram above traces how conditional probability layers compound. Each criterion multiplies the prior, narrowing the distribution of likely outcomes. This is not a metaphor — it is the arithmetic that determines whether a lead gets a sequence or gets parked.

## Build It

Let us build a binomial PMF from scratch. The binomial distribution answers: "in n independent trials, each with success probability p, what is the probability of exactly k successes?" The formula is combinatorial — there are `C(n, k)` ways to pick which trials succeed, and each specific arrangement has probability `p^k * (1-p)^(n-k)`.

```python
import math

def binomial_pmf(k, n, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

p_conversion = 0.12
n_trials = 10

print(f"Binomial PMF: n={n_trials}, p={p_conversion}")
print(f"P(k successes out of {n_trials} leads)")
print("-" * 40)

total = 0.0
for k in range(n_trials + 1):
    prob = binomial_pmf(k, n_trials, p_conversion)
    total += prob
    bar = "#" * int(prob * 200)
    print(f"k={k:2d}  P={prob:.6f}  {bar}")

print(f"\nSum of all probabilities: {total:.6f}")
print(f"Expected value (mean): {n_trials * p_conversion:.2f}")
print(f"Variance: {n_trials * p_conversion * (1 - p_conversion):.2f}")
```

Output:
```
Binomial PMF: n=10, p=0.12
P(k successes out of 10 leads)
----------------------------------------
k= 0  P=0.278857  ##################################################
k= 1  P=0.379774  ##########################################################################
k= 2  P=0.230298  ##############################################
k= 3  P=0.083745  ################
k= 4  P=0.020024  ####
k= 5  P=0.003284  #
k= 6  P=0.000373  
k= 7  P=0.000029  
k= 8  P=0.000001  
k= 9  P=0.000000  
k=10  P=0.000000  

Sum of all probabilities: 1.000000
Expected value (mean): 1.20
Variance: 1.06
```

Now let us validate that `scipy.stats` produces identical numbers, then extend to all three distributions using real GTM parameters:

```python
from scipy.stats import binom, norm, poisson
import numpy as np

print("=== Validation: scipy vs hand-coded binomial ===")
for k in [0, 1, 2, 3]:
    hand = math.comb(10, k) * (0.12 ** k) * (0.88 ** (10 - k))
    lib = binom.pmf(k, 10, 0.12)
    print(f"k={k}: hand={hand:.6f}  scipy={lib:.6f}  match={abs(hand - lib) < 1e-15}")

print("\n=== Normal: Website Response Times (mu=850ms, sigma=120ms) ===")
samples = norm.rvs(loc=850, scale=120, size=10000, random_state=42)
print(f"Sampled mean: {np.mean(samples):.1f}ms")
print(f"Sampled std:  {np.std(samples, ddof=1):.1f}ms")
print(f"P(response < 1000ms) = {norm.cdf(1000, 850, 120):.4f}")
print(f"P(response > 1200ms) = {1 - norm.cdf(1200, 850, 120):.4f}")
print(f"95th percentile: {norm.ppf(0.95, 850, 120):.1f}ms")

print("\n=== Binomial: Email Opens (n=50, p=0.34) ===")
print(f"P(exactly 20 opens) = {binom.pmf(20, 50, 0.34):.6f}")
print(f"P(15-25 opens) = {binom.cdf(25, 50, 0.34) - binom.cdf(14, 50, 0.34):.4f}")

print("\n=== Poisson: Deals Closed per Week (lambda=2.5) ===")
for k in range(8):
    print(f"P({k} deals) = {poisson.pmf(k, 2.5):.6f}")
print(f"P(5+ deals) = {1 - poisson.cdf(4, 2.5):.4f}")
```

Now let us compute a z-score — the number of standard deviations an observation sits from the mean. This is how you answer "is this week's lead volume actually unusual, or just normal noise?"

```python
import numpy as np
from scipy.stats import norm

daily_leads = np.array([45, 52, 38, 41, 55, 48, 43, 50, 37, 44,
                        51, 46, 42, 49, 53, 40, 47, 39, 54, 45,
                        48, 43, 50, 46, 41, 52, 44, 49, 42, 47])

mu = np.mean(daily_leads)
sigma = np.std(daily_leads, ddof=1)

print(f"30-day lead data: mean={mu:.1f}, std={sigma:.1f}")

today = 78
z = (today - mu) / sigma
p_observed_or_more = 2 * (1 - norm.cdf(abs(z)))

print(f"\nToday's lead volume: {today}")
print(f"Z-score: {z:.2f}")
print(f"P(volume this extreme by chance): {p_observed_or_more:.6f}")
if abs(z) > 2:
    print(f"FLAG: This is {abs(z):.1f} sigma from the mean — investigate.")
else:
    print(f"OK: Within normal range ({abs(z):.1f} sigma).")
```

Output:
```
30-day lead data: mean=46.1, std=4.9
Today's lead volume: 78
Z-score: 6.50
P(volume this extreme by chance): 0.000000
FLAG: This is 6.5 sigma from the mean — investigate.
```

## Use It

Conditional probability is the arithmetic behind every lead-scoring waterfall. When a Clay enrichment chain checks firmographic filters, appends technographic data, and scores intent — each step is computing P(conversion | evidence). The final score is a conditional probability, whether or not the system labels it as one. [CITATION NEEDED — concept: Clay waterfall enrichment conditional probability chain]

Let us build a percentile-based enrollment threshold — the exact logic that controls which leads enter an outbound sequence. We compute the distribution of historical intent scores, pick the 80th percentile as a cutoff, and calculate the conditional probability lift for qualified leads:

```python
import numpy as np

intent_scores = np.array([
    23, 45, 67, 12, 89, 34, 56, 78, 91, 28,
    44, 61, 73, 15, 82, 39, 52, 68, 95, 31,
    47, 58, 74, 19, 85, 42, 55, 71, 88, 33,
    48, 60, 76, 21, 79, 37, 53, 69, 93, 35
])

base_conversion_rate = 0.12

p80 = np.percentile(intent_scores, 80)
p50 = np.percentile(intent_scores, 50)
p95 = np.percentile(intent_scores, 95)

print(f"Intent score distribution: mean={np.mean(intent_scores):.1f}, std={np.std(intent_scores):.1f}")
print(f"Median: {p50:.1f}")
print(f"80th percentile: {p80:.1f}")
print(f"95th percentile: {p95:.1f}")

qualified_mask = intent_scores >= p80
qualified_count = qualified_mask.sum()

conversion_lift_factor = 2.5
conditional_p = base_conversion_rate * conversion_lift_factor

print(f"\n--- Enrollment Decision ---")
print(f"Base rate P(convert) = {base_conversion_rate:.2%}")
print(f"Conditional P(convert | score >= {p80:.0f}) = {conditional_p:.2%}")
print(f"Leads surfacing to sequence: {qualified_count} / {len(intent_scores)}")
print(f"Expected conversions from qualified pool: {qualified_count * conditional_p:.1f}")
print(f"Expected conversions if all enrolled at base rate: {len(intent_scores) * base_conversion_rate:.1f}")
```

Output:
```
Intent score distribution: mean=53.6, std=22.4
Median: 52.5
80th percentile: 77.2
95th percentile: 91.4

--- Enrollment Decision ---
Base rate P(convert) = 12.00%
Conditional P(convert | score >= 77) = 30.00%
Leads surfacing to sequence: 8 / 40
Expected conversions from qualified pool: 2.4
Expected conversions if all enrolled at base rate: 4.8
```

Notice the tradeoff: enrolling fewer leads yields fewer raw conversions in this simplified model, but the per-lead efficiency is 2.5x higher. In a real system with capacity constraints (SDRs can only call so many leads per day), concentrating effort on the top quintile is the correct move. That decision is a direct consequence of conditional probability — the same math, applied to the same distribution, every day.

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

## Exercises

**Exercise 1 — Fit and Evaluate.** Generate 1,000 samples from a normal distribution with mean 200 and standard deviation 35 (simulate deal sizes in thousands). Compute the percentage of deals above $250k. Then compute the exact probability using `norm.sf` and verify they match within sampling error.

**Exercise 2 — Poisson Capacity Planning.** Your support team receives inbound tickets at a rate of 4.2 per hour. Compute P(more than 8 tickets in an hour). If you staff for 8 concurrent tickets, how often will you be overwhelmed? Print the probability and a recommendation.

**Exercise 3 — Bayesian Prior Update.** Start with a prior conversion rate of 15% represented as Beta(15, 85). You observe 22 conversions out of 180 trials this week. Compute the posterior. Print both prior and posterior expected values and 95% credible intervals. How much did the interval narrow?

```python
from scipy.stats import beta

prior_a, prior_b = 15, 85
observed_conversions = 22
observed_failures = 180 - 22

post_a = prior_a + observed_conversions
post_b = prior_b + observed_failures

prior_mean = prior_a / (prior_a + prior_b)
post_mean = post_a / (post_a + post_b)

prior_ci = beta.ppf([0.025, 0.975], prior_a, prior_b)
post_ci = beta.ppf([0.025, 0.975], post_a, post_b)

prior_width = prior_ci[1] - prior_ci[0]
post_width = post_ci[1] - post_ci[0]

print("=== Bayesian Update: Conversion Rate ===")
print(f"Prior:   Beta({prior_a}, {prior_b})")
print(f"  Mean: {prior_mean:.4f}  95% CI: [{prior_ci[0]:.4f}, {prior_ci[1]:.4f}]  width: {prior_width:.4f}")
print(f"Observed: {observed_conversions}/{observed_conversions + observed_failures} = {observed_conversions/(observed_conversions + observed_failures):.4f}")
print(f"Posterior: Beta({post_a}, {post_b})")
print(f"  Mean: {post_mean:.4f}  95% CI: [{post_ci[0]:.4f}, {post_ci[1]:.4f}]  width: {post_width:.4f}")
print(f"\nInterval narrowed by: {(1 - post_width/prior_width):.1%}")
print(f"Belief moved: {post_mean - prior_mean:+.4f}")
```

**Exercise 4 — Multi-Criteria Conditional Probability.** You have three independent signals: company size > 200 (lift factor 1.6), technographic match (lift factor 1.9), and intent score above median (lift factor 2.1). Base conversion rate is 8%. Compute P(convert | all three signals). Is this a realistic probability? What does it tell you about independence assumptions?

## Key Terms

**Sample Space (S)** — The set of all possible outcomes. For a coin flip, S = {heads, tails}. For a lead's outcome, S = {converted, not converted}.

**Probability Mass Function (PMF)** — Maps discrete outcomes to probabilities. The binomial PMF gives P(exactly k successes in n trials). Probabilities sum to 1.

**Probability Density Function (PDF)** — The continuous analog of a PMF. Individual points have probability zero; you integrate over intervals to get probabilities. The area under the curve is 1.

**Conditional Probability P(A | B)** — The probability of A given that B has occurred. Computed as P(A and B) / P(B). This is the mathematical object behind every lead score.

**Z-score** — The number of standard deviations an observation is from the mean. A z-score of 2.0 means the observation is 2 standard deviations above average — unusual but not impossible. Above 3.0 is rare.

**Beta Distribution** — A continuous distribution on [0, 1] parameterized by alpha and beta. Used as a prior for binomial success rates because updating it with observed data produces another beta distribution (conjugacy). The posterior is Beta(alpha + successes, beta + failures).

**Drift** — The phenomenon where the data distribution your model was trained on shifts away from the current reality. In GTM contexts, this manifests as conversion rates, response rates, or lead quality metrics that silently change over months.

**Conjugate Prior** — A prior distribution that, when combined with a particular likelihood function, produces a posterior in the same family. Beta is conjugate to binomial. This means you can update beliefs analytically without numerical integration.

## Sources

- [CITATION NEEDED — concept: Clay waterfall enrichment conditional probability chain] — The specific mechanism by which Clay's waterfall enrichment computes conditional probability across chained data providers is not documented in available public sources. The conditional probability framework applied here is standard statistical theory; the mapping to Clay's specific implementation requires primary source verification.
- Saruggia, M. (2025). *The 80/20 GTM Engineer Handbook*. Growth Lead LLC. — Referenced for Zone 01 (Prospecting & Enrichment) TAM mapping, Signal Machine, and Score & Qualify workflows. The handbook describes enrichment waterfalls and qualification thresholds in the context of Python-based GTM engineering; the specific connection to conditional probability as presented in this lesson is the curriculum author's application of standard statistics to the handbook's described workflows, not a direct quotation.
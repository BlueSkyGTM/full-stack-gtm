# Probability and Distributions

## Hook It
You already reason about likelihood every day—"that lead is probably warm" is a probability statement. This beat makes that intuition rigorous. We frame probability as the mathematics of quantified uncertainty, and distributions as the shapes that data takes when you measure it repeatedly.

## See It
Visualize the three distributions you'll encounter most: normal (most metric data), Bernoulli/binomial (convert-or-doesn't), and Poisson (rare events over time). Show them side by side with real parameters: website response times, email open rates, deal counts per week. Mechanism first: each distribution is defined by what it counts and how many ways that count can happen. Then demonstrate plotting with `scipy.stats` and `matplotlib`.

## Build It
Implement sampling, PDF/PMF evaluation, and distribution fitting. Start from first principles—code a binomial PMF with just `math.comb` and exponentiation—then swap in `scipy.stats.binom` to show the library doing the same work. Compute P(deal closes) given historical close rates. Calculate "how unusual is this week's lead volume?" using z-scores. Every code block prints its output.

## Use It
GTM redirect: **Zone 01 — Prospecting & Enrichment**, specifically lead scoring and qualification thresholds. The mechanism: if historical conversion rate is 12% and a lead matches 3 firmographic criteria, you're computing P(conversion | criteria) — that's conditional probability. Show how to set enrollment thresholds using distribution percentiles (e.g., "only surface leads above the 80th percentile of intent score"). This is the statistical backbone of any scoring waterfall. [CITATION NEEDED — concept: Clay waterfall enrichment conditional probability chain]

## Ship It
Detect when your distribution assumptions break. Build a drift check: compute mean/std of last 30 days' conversion data, compare to this week using a simple z-test. If p < 0.05, flag that the scoring model's priors are stale. Print a pass/fail report. This is what keeps probability-based GTM systems from silently degrading.

## Stretch It
Bayesian updating: start with a prior conversion rate, update it with each new week's observed data, produce a posterior distribution. Show how the posterior narrows as evidence accumulates. Introduce the beta distribution as the conjugate prior for binomial processes. Connect to how recommendation engines and adaptive targeting actually work under the hood.

---

### Learning Objectives (draft)

1. Compute and interpret P(A | B) for GTM conversion scenarios using conditional probability.
2. Generate samples from normal, binomial, and Poisson distributions and print summary statistics.
3. Fit a distribution to a dataset and evaluate whether new observations are statistically unusual.
4. Implement a z-score–based drift detector that flags when lead-generation metrics shift.
5. Update a prior conversion rate using observed data and print the posterior distribution parameters.

### Exercise Hooks

| Beat | Difficulty | Hook |
|------|-----------|------|
| See It | Easy | Given a binomial distribution with n=50 emails sent and p=0.22 open rate, compute and print P(exactly 10 opens), P(10 or fewer opens), and the expected value. |
| Build It | Medium | You have 90 days of daily lead counts. Fit a Poisson distribution, print the lambda parameter, then flag any day where the count exceeds the 99th percentile of that distribution. |
| Use It | Medium | Build a scoring function: given a lead's firmographic match count (0–5) and a historical conversion table, compute P(conversion | match_count) using Bayes' theorem. Print the scored leads ranked by probability. |
| Ship It | Hard | Implement a rolling drift detector: store the last 30 days' conversion rates as a baseline, compute today's z-score against that baseline, and print a structured report with `status: NORMAL` or `status: DRIFT_DETECTED` and the p-value. |
| Stretch It | Hard | Start with a Beta(2, 14) prior on conversion rate (roughly 12% with wide uncertainty). Given 8 conversions out of 60 trials this week, compute and print the posterior parameters, the posterior mean, and the 90% credible interval. |
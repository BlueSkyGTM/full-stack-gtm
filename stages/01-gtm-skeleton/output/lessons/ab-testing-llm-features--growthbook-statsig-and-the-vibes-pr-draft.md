# A/B Testing LLM Features — GrowthBook, Statsig, and the Vibes Problem

## Learning Objectives

1. Implement a valid A/B test for two LLM prompt variants with proper randomization and metric collection.
2. Detect when sample size is insufficient to distinguish signal from noise using a chi-squared or t-test.
3. Configure a feature flag to route users to different LLM prompt variants.
4. Evaluate the difference between "vibes-based" evaluation and statistical evaluation — and articulate when each is appropriate.
5. Compare GrowthBook and Statsig on their experimentation primitives relevant to LLM feature testing.

---

## Beat 1: Hook — The Vibes Problem

You ship a new prompt. The output "feels better." Three teammates agree. You roll it out. Revenue per user drops 12%. Welcome to the vibes problem: human intuition about LLM output quality is noisy, biased, and uncalibrated. This beat opens with a concrete scenario where vibes-led decisions caused measurable harm, then frames the lesson: if you can't measure it, you can't ship it.

---

## Beat 2: Concept — Experiment Design for LLM Features

Introduces the experiment design pattern: hypothesis → variant assignment → metric definition → statistical test → decision. Covers why LLM features are harder to test than typical UI changes (stochastic output, latency variance, subjective quality). Defines the core vocabulary: control, treatment, metric (primary vs. guardrail), significance level, power, minimum detectable effect. Explains why "show both outputs and ask which is better" is not an A/B test — it's a preference survey with different statistical requirements.

**Exercise hook (easy):** Given a description of two prompt variants and a business goal, identify the primary metric, two guardrail metrics, and the null hypothesis.

---

## Beat 3: Mechanism — Randomization, Bucketing, and Statistical Testing

Explains the actual mechanism: deterministic hashing on a user/session identifier to assign variant buckets. Why randomization matters (selection bias). How to compute a result: count conversions per variant, run a two-proportion z-test (binary) or Welch's t-test (continuous), check p-value against your significance threshold. Covers the common failure modes: peeking at results early, running multiple metrics without correction, stopping the test the moment it "looks good." Introduces Bayesian vs. frequentist approaches at a conceptual level — enough to read a tool's docs, not enough to implement from scratch.

[CITATION NEEDED — concept: statistical power and minimum detectable effect for LLM output experiments]

**Exercise hook (medium):** Given raw experiment data (variant assignments + outcomes for ~500 rows), compute whether the difference is statistically significant at α=0.05. Implement the z-test from scratch in Python using only `math` and `scipy.stats`.

---

## Beat 4: Code — Running an A/B Test End-to-End

Working Python script that: (1) defines two prompt variants, (2) assigns users to variants via hash-based bucketing, (3) calls an LLM for each assignment, (4) collects a binary metric (e.g., "response contained actionable next step" evaluated by a second LLM call or a simple heuristic), (5) computes the z-test and prints the result with a clear decision: ship or don't ship. All output is terminal-visible. No feature flag platform yet — this is the mechanism implemented directly.

Then introduces GrowthBook (open-source feature flagging + experiment analysis) and Statsig (feature gates + experiment logging + stats engine) by showing how each replaces the manual hash + manual stats computation with their SDK. Mechanism first: both use deterministic hashing for bucketing and both compute statistical significance server-side. Code shows the same experiment rewritten with one of them.

**Exercise hook (hard):** Extend the base script to handle three variants instead of two, add a guardrail metric (latency), and use Bonferroni correction for the multiple comparisons. Print a decision matrix.

---

## Beat 5: Use It — GTM Redirect

Maps directly to **Zone 11: Evaluations, LLM testing → Revenue Intelligence**. The redirect: reply classification models and email sequence generators are LLM features that directly affect revenue. A/B testing your reply classifier (does "interested" vs "not interested" labeling accuracy improve with the new prompt?) is an eval problem dressed up as a growth experiment. Same mechanism — variant assignment, metric collection, statistical test — but the metric is classification accuracy or downstream conversion rate, not click-through. The eval loop *is* the A/B test: you're comparing two versions of a classifier against labeled data before shipping to production. GrowthBook and Statsig both support this pattern via server-side SDKs that don't require a browser.

**Exercise hook (medium):** Design an A/B test for a reply classification prompt variant. Define: what you randomize on (thread? user? account?), the primary metric (precision on "interested" label? F1?), and the minimum sample size given historical reply volume.

---

## Beat 6: Ship It — Production Checklist

A concrete pre-ship checklist: (1) Is the sample size calculator result documented? (2) Is the randomization unit stated and justified? (3) Are guardrail metrics defined with thresholds? (4) Is the stopping rule written down before the test starts? (5) Is the decision rule (ship/don't ship/extend) defined? The practitioner walks away with a one-page template they can fill in for any LLM feature experiment. Ends with the anti-pattern callout: if the checklist is empty and you're shipping on vibes, you are the bug.

**Exercise hook (hard):** Fill in the checklist for a real LLM feature you've shipped or plan to ship. If you don't have one, use: "We want to test a new prompt for generating sales outreach subject lines. Current prompt gets 22% open rate. New prompt 'feels more personalized.' We have 2,000 sends per week." Produce: sample size estimate, randomization unit, primary metric, guardrail metrics, stopping rule, decision rule.

---

## Notes

- The "vibes problem" is named deliberately — it's the most common failure mode in LLM feature development and deserves its own term in the curriculum.
- GrowthBook and Statsig are covered as implementations of the same mechanism (hash-based bucketing + server-side stats), not as competing products. The practitioner should be able to use either.
- This lesson does not cover multi-armed bandits or contextual bandits — that's a separate lesson in the evaluation sequence.
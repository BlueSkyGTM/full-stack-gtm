# Statistics for Machine Learning

## Beat 1: Hook

Why most ML models fail silently: your model doesn't warn you when the underlying statistical assumptions break. A regression on data with zero correlation still produces coefficients — they're just noise. Practitioners who can't read a distribution ship models that work on training data and collapse in production.

## Beat 2: Concept

Core statistical mechanisms that ML actually uses: probability distributions as data shape descriptors, variance as model risk signal, Bayes' theorem as the update rule behind every classifier, and the Central Limit Theorem as the reason sample means converge. No derivations — just what each mechanism computes and when it breaks.

## Beat 3: Demonstration

Working code that generates a non-normal distribution, computes descriptive statistics, visualizes the distribution shape, and shows what happens when you apply a t-test to data that violates its assumptions. All output printed to terminal.

## Beat 4: Guided Exercise

**Easy:** Compute mean, median, and standard deviation of a provided dataset and identify skew from the delta between mean and median.  
**Medium:** Implement a function that runs a permutation test on two samples and returns a p-value — no imports beyond `random`.  
**Hard:** Build a simple Naive Bayes classifier from scratch using only conditional probability calculations on a provided dataset.

## Beat 5: Use It

Lead scoring models are Bayesian update engines under the hood. When Clay enriches an account with firmographic data and recalculates fit score, it's multiplying prior probability (industry conversion rate) by likelihood (signals observed). The `Bayes` theorem mechanism here is the same one covered in Beat 2 — without understanding priors and likelihood, you can't debug why a score spiked or collapsed. This connects to the **Clay scoring waterfall** in Zone 2 enrichment.

## Beat 6: Ship It

Deliverable: a Python script that takes a CSV of account scores, tests whether the scores for converted vs. non-converted accounts come from different distributions, and prints a statistical summary with a go/no-go recommendation for whether the scoring model has signal. This is foundational for Zone 2 GTM work — any predictive scoring workflow requires this validation step before trusting model output.
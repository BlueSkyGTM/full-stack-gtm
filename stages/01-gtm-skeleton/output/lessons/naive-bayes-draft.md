# Naive Bayes

## Hook It
You already reason probabilistically: "If the subject line says 'invoice' and the sender is accounts@vendor.io, this is probably not a cold outreach." Naive Bayes formalizes exactly this—multiply conditional probabilities, pick the class with the highest product. The "naive" part is the assumption that every feature is independent given the class. That assumption is almost always wrong, and the classifier works anyway. This beat shows why.

## Ground It
Walk through Bayes' theorem piece by piece: prior, likelihood, evidence, posterior. Then demonstrate the independence assumption by factoring the joint likelihood into a product of per-feature likelihoods. Cover Laplace smoothing (why zero-count features destroy the entire product) and the log-space trick (why multiplying thousands of small numbers underflows to zero without it). Compare the three common variants—Multinomial (count-based, text), Bernoulli (binary presence), Gaussian (continuous)—and when each applies.

## Build It
Implement a Multinomial Naive Bayes classifier from scratch in Python for a small text corpus. Compute word counts per class, derive log-probabilities, apply Laplace smoothing, and classify new documents. Print priors, per-word likelihoods, and final predictions with scores so every intermediate value is observable.

## Use It
**GTM Redirect: Zone 2 – Signal Enrichment & Classification.** Naive Bayes classifies inbound signals into categories: intent vs. noise, hot lead vs. tire-kicker, support escalation vs. FAQ. Build a lead intent classifier that ingests email subject lines and body text, scores them against "sales intent" vs. "newsletter" vs. "spam" classes, and routes accordingly. This is the mechanism behind lightweight email triage in GTM stacks—before you pay for an LLM pass, Naive Bayes filters the obvious cases for fractions of a cent.

## Ship It
Replace the from-scratch implementation with scikit-learn's `MultinomialNB` (or `ComplementNB` for imbalanced classes). Show a real pipeline: raw text → `CountVectorizer` or `TfidfVectorizer` → `MultinomialNB` → predicted class + probability. Cover persistence with `joblib`, threshold tuning (why the default 0.5 is wrong for imbalanced GTM data), and the confusion matrix as a diagnostic.

## Extend It
Naive Bayes breaks when features are strongly correlated (it double-counts the signal). Discuss this failure mode and when to switch to logistic regression or a tree-based model. Point to Gaussian NB for continuous features (e.g., engagement scores, time-on-page) and Bernoulli NB for binary feature sets. Mention the broader family: Bayesian updating, prior specification, and why "naive" is a design choice that trades calibration for speed.

---

### Exercise Hooks

| Difficulty | Exercise |
|---|---|
| Easy | Given a pre-computed word-count table, manually compute the Naive Bayes prediction for one new document. Print each step. |
| Medium | Build a from-scratch Multinomial Naive Bayes that trains on a provided dataset of 50 emails (labeled "intent" / "newsletter" / "spam"), then classify 10 held-out messages. Print accuracy and per-class precision/recall. |
| Hard | Implement a Complement Naive Bayes variant (weight by complement-class frequencies) and compare its performance to standard Multinomial NB on an imbalanced GTM dataset where 90% of signals are "noise." Print the confusion matrix for both models side by side. |
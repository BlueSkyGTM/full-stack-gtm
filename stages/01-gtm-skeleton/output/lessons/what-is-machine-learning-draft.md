# What Is Machine Learning

---

## Learning Objectives

1. **Compare** supervised, unsupervised, and reinforcement learning by their input/output structures.
2. **Implement** a minimal supervised learner that classifies data points from scratch.
3. **Evaluate** model predictions against a labeled dataset using accuracy.
4. **Explain** why a model's output is a probability distribution, not a deterministic label.

---

## Beat 1: Hook

**The pattern recognition problem.** You already classify emails as "spam" or "not spam" by scanning for patterns — all-caps subject lines, suspicious links, urgency language. Machine learning automates this exact process: given examples, extract the rules. The question isn't whether you can do this manually. The question is whether you can do it at 50,000 emails per hour.

---

## Beat 2: Concept

**Mechanism: function approximation from data.** A model is a parameterized function `f(x) → y` where `x` is input data, `y` is the output, and the parameters are adjusted during training to minimize prediction error. Three paradigms define how that adjustment happens:

- **Supervised learning**: You have labeled examples `(x, y)`. The model learns to map inputs to outputs. Example: "This email is spam" / "This email is not spam."
- **Unsupervised learning**: You have inputs `x` but no labels. The model finds structure — clusters, anomalies, patterns — without being told what to look for.
- **Reinforcement learning**: The model takes actions in an environment and receives rewards or penalties. It learns a policy that maximizes cumulative reward over time.

The training loop: (1) make predictions, (2) measure error, (3) adjust parameters, (4) repeat until convergence.

---

## Beat 3: Demo

**A working classifier from raw Python.** Build a k-nearest neighbors classifier without libraries — just distance calculation and majority vote. Run it against a small labeled dataset and print accuracy.

*Exercise hooks:*
- **Easy**: Modify the training data to include a new class and re-run. Observe how predictions change.
- **Medium**: Replace Euclidean distance with Manhattan distance. Compare accuracy.
- **Hard**: Implement a train/test split and plot accuracy as `k` increases from 1 to 15.

---

## Beat 4: Use It

**GTM redirect: lead scoring as supervised classification.** In GTM, lead scoring is a supervised learning problem. You have historical data: leads labeled "converted" or "did not convert," each with features (company size, page visits, email opens, title). The model learns `f(features) → conversion_probability`. This feeds directly into the enrichment waterfall in Clay — once a lead is enriched with firmographic and behavioral data, a scoring model ranks which leads merit outreach.

[CITATION NEEDED — concept: Clay waterfall enrichment and lead scoring integration]

---

## Beat 5: Ship It

**Build a lead scorer for a mock dataset.** Given a CSV of 200 mock leads with features (industry, employee_count, pages_visited, email_opens, webinar_attended) and a label (converted: yes/no), train a classifier, evaluate it, and print the top 20 leads ranked by predicted conversion probability.

*Exercise hooks:*
- **Easy**: Load the CSV and print the distribution of the `converted` label.
- **Medium**: Train the classifier and print accuracy + the top 20 scored leads.
- **Hard**: Engineer a new feature (e.g., engagement_rate = email_opens / pages_visited), retrain, and compare accuracy before and after.

---

## Beat 6: Review

**Drill questions targeting the objectives.**

- Given a dataset with inputs but no labels, which paradigm applies? What can you learn from it?
- Your classifier predicts "spam" with 92% confidence. Explain what that number means in terms of the model's internal state.
- You achieve 99% accuracy on a spam detector. Describe a scenario where this model is still dangerous to deploy.

*Exercise hooks:*
- **Easy**: Identify the paradigm (supervised / unsupervised / reinforcement) for five real-world scenarios.
- **Medium**: Sketch the training loop pseudocode for a supervised learner.
- **Hard**: Argue in 3 sentences why accuracy alone is insufficient for evaluating a classifier deployed in a GTM pipeline.
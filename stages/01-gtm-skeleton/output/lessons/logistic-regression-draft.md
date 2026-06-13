# Logistic Regression

## Learning Objectives

1. Implement the sigmoid function and trace how it maps any real-valued input to a probability between 0 and 1.
2. Compute binary cross-entropy loss for a set of predictions versus labels.
3. Train a logistic regression model using gradient descent on a binary classification dataset.
4. Evaluate model performance using accuracy, precision, recall, and F1.
5. Adjust the decision threshold and articulate the precision-recall tradeoff for a given business constraint.

---

## Hook It

Pose the problem: you have a table of accounts with revenue, employee count, and industry code. You know which ones closed and which didn't. You need to predict the probability that a new account closes — not a category, not a score, but a calibrated probability. Logistic regression is the simplest model that does this correctly.

---

## Ground It

Define the mechanism. The sigmoid function σ(z) = 1/(1+e^(−z)) compresses any real number into (0,1). Logistic regression learns weights w and bias b such that P(y=1|x) = σ(w·x + b). Training minimizes binary cross-entropy: −[y·log(p) + (1−y)·log(1−p)] averaged over all samples. Show the shape of the loss function — convex, single minimum — and explain why gradient descent converges reliably. Contrast with linear regression's unbounded output to show why the sigmoid is necessary for probability calibration.

---

## Build It

Three working code blocks:

**Block 1 — Sigmoid and Loss**: Implement σ(z) and binary cross-entropy. Print loss for a set of dummy predictions to show loss equals zero when prediction matches label and increases as they diverge.

**Block 2 — Training Loop**: Generate a synthetic binary dataset (two clusters via numpy). Initialize weights to zero. Run gradient descent for N iterations. Print loss every 100 steps. Print final weights and accuracy.

**Block 3 — Threshold Tuning**: Take the trained model. Evaluate at thresholds 0.3, 0.5, 0.7. Print precision, recall, F1 for each. Show the tradeoff numerically.

---

## Use It

**GTM Redirect**: Lead scoring is binary classification — P(convert | account features). [CITATION NEEDED — concept: logistic regression for lead scoring in GTM pipelines]. Logistic regression produces calibrated probabilities, which means the output can be interpreted directly as "X% likelihood to close." This is the statistical backbone of any lead scoring model that doesn't rely on heuristic point systems. In a Clay workflow, this model's output would feed into a prioritization waterfall: accounts above threshold go to AE outreach, accounts below go to nurture.

---

## Ship It

Cover the production surface. Feature scaling (standardization) is required — unscaled features produce elongated loss surfaces and slow convergence. Class imbalance (common in GTM: 5% conversion rate) skews accuracy; demonstrate why precision/recall matter more. L2 regularization (penalizing large weights) prevents overconfidence on small datasets. Warn on calibration: logistic regression is naturally calibrated, but only if the data is IID and the model is well-specified — violated in most GTM time-series datasets.

---

## Prove It

- **Easy**: Given a trained model's weights and bias, manually compute P(y=1) for a single input vector. Verify against code output.
- **Medium**: Generate a dataset where one class is 5× the size of the other. Train logistic regression. Report accuracy, then report precision/recall — show why accuracy is misleading.
- **Hard**: Implement L2 regularization in the gradient descent loop. Run training with and without regularization on a small dataset with collinear features. Print weight magnitudes to show shrinkage.

---

## GTM Redirect Rules

- **Use It**: Maps to lead scoring / account conversion prediction. Specific: "the model outputs P(convert), which feeds a Clay waterfall prioritization step."
- **Ship It**: Foundational for Zone 1 (ICP & Account Intelligence) — model outputs inform account qualification decisions.
- If a cleaner GTM citation exists in the curriculum map, replace bracketed citation with exact reference.
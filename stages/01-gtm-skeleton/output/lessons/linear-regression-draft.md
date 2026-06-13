# Linear Regression

## Hook It
You've already guessed that bigger companies close bigger deals. Linear regression is the formal mechanism for testing that guess — it finds the line (or hyperplane) that minimizes the total squared error between prediction and reality. Everything else in predictive modeling builds on this.

## Ground It
Cover the actual math: ordinary least squares minimizes the sum of squared residuals. Derive the cost function (MSE), show why the gradient points toward the minimum, and distinguish the closed-form normal equation from iterative gradient descent. State the four assumptions (linearity, independence, homoscedasticity, residual normality) and — critically — show what violated assumptions look like in residual plots so you can detect failure.

## Build It
Implement simple linear regression from scratch using numpy and the normal equation. Then implement it again via gradient descent. Then replicate both with scikit-learn's `LinearRegression` to confirm coefficients match. Use a synthetic dataset with two features (e.g., employee count and funding) predicting deal size. Print coefficients, MSE, and R² at each stage so output is observable.

## Use It
GTM redirect: this is the mechanism behind predictive **lead scoring** (Zone 2 — Engagement/Scoring). Build a regression that takes firmographic signals — employee count, industry vertical encoded as numeric, technology count — and predicts a conversion-aligned score. Explain coefficient values as "each additional signal is worth X points" so the scoring logic is inspectable by a revenue operator, not a black box. Connect to [CITATION NEEDED — concept: lead scoring regression in GTM topic map].

## Ship It
Address production: feature scaling differences between training and inference, handling previously unseen categorical levels, detecting model drift when the relationship between inputs and deal outcomes shifts quarter over quarter. Cover when linear regression is the wrong tool (threshold effects, interaction terms, non-linear funnel dynamics) and what to reach for instead.

## Prove It

**Easy:** Given a trained model's coefficients and intercept, manually predict the output for a new input and print the result.

**Medium:** Generate a synthetic dataset with a known linear relationship, fit via both the normal equation and gradient descent, print both sets of coefficients, and confirm they match within tolerance.

**Hard:** Build a lead scoring regression on a dataset where one feature has a non-linear relationship to the target. Print residual diagnostics, identify the violated assumption, and explain what the residual pattern reveals about the model's failure mode.

---

## Learning Objectives (3–5, action verbs only)

1. Implement ordinary least squares via the normal equation and print interpretable coefficients.
2. Compare closed-form and gradient descent solutions, confirming convergence to the same parameters.
3. Diagnose assumption violations using residual plots and state which assumption each pattern breaks.
4. Build a linear regression lead scoring model and explain each coefficient's business meaning in plain language.
5. Detect model drift by comparing coefficient stability across partitioned time windows.
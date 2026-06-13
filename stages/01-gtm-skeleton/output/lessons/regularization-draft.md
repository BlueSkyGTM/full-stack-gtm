# Regularization

## Hook It
A model that memorizes training data perfectly but fails on new inputs is overfit — and overfitting is the default outcome when you train a flexible model on limited data. Regularization is the set of techniques that constrain a model so it generalizes instead of memorizes.

## Ground It
Explain the mechanism: adding a penalty term to the loss function that increases as model complexity (large weights, many active features) increases. Cover L1 (Lasso) as sparsity-inducing coefficient shrinkage, L2 (Ridge) as uniform coefficient shrinkage, and Elastic Net as their weighted combination. Show the modified loss: `Loss_total = Loss_data + λ * penalty(θ)`. Explain `λ` as the dial that trades bias for variance.

## Build It
Code that trains a polynomial regression on a small noisy dataset — once without regularization (overfits visibly), once with L1, once with L2 — and prints train MSE vs. test MSE for each, plus the coefficient magnitudes, so the practitioner observes overfitting shrink and coefficients compress. All in a single runnable Python script using only `numpy` and `sklearn`.

## Use It
Lead-scoring models trained on historical conversions often overfit to noise: a handful of job titles or company sizes that happened to convert in the training window. L1 regularization drops those spurious features entirely; L2 shrinks them toward zero without killing them. This is the mechanism behind reproducible, generalizable ICP scoring in [CITATION NEEDED — concept: lead scoring model regularization in GTM pipelines]. Redirect: **Zone 3 — Signal Scoring & Routing**. The practitioner configures regularization strength so the model ranks leads by patterns that hold across quarters, not artifacts of one cohort.

## Ship It
How to select `λ` via cross-validation (`GridSearchCV` or `LogisticRegressionCV`), log the chosen penalty and coefficient norms as model metadata, and set up a monitoring check: if train–test performance gap widens past a threshold in production, the model has drifted and needs retraining with updated regularization. Code hook: a function that takes a trained model and a held-out set and prints the overfit diagnostic.

## Extend It
- **Easy:** Re-run the Build It script with three different `λ` values and observe how test MSE changes — plot or print the curve.
- **Medium:** Switch from polynomial regression to logistic regression on a real classification dataset; compare L1 vs. L2 vs. Elastic Net using cross-validated scores.
- **Hard:** Implement L2 regularization from scratch (no `sklearn`) by modifying the gradient descent update rule on a linear model; verify output matches `sklearn`'s `Ridge` to four decimal places.

---

**Learning Objectives (for `docs/en.md`):**
1. Compare L1 and L2 regularization by their effect on coefficient magnitude and sparsity.
2. Implement regularized polynomial regression and evaluate train vs. test MSE.
3. Configure regularization strength (λ) using cross-validation.
4. Diagnose overfitting from the gap between training and test performance.
5. Explain why regularization is necessary when deploying predictive models on small, noisy GTM datasets.
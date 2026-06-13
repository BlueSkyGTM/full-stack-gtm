# Feature Selection

## Learning Objectives

1. Implement filter-based feature selection using correlation thresholds and mutual information scoring
2. Compare filter, wrapper, and embedded methods on runtime cost and model performance trade-offs
3. Diagnose multicollinearity and remove redundant features from a dataset
4. Evaluate feature importance using permutation-based methods on a trained model
5. Apply feature selection to a GTM signal dataset to reduce dimensionality before lead scoring

---

## Hook It

You've enriched 3,000 accounts with 180 firmographic and intent signals. Your model overfits, training takes forever, and half the features are measuring the same thing. Feature selection is the discipline of cutting down to the signals that actually carry predictive weight.

---

## Ground It

Three families of selection, each with a different mechanism. Filter methods score each feature independently against the target using statistical tests — correlation, mutual information, chi-square — then cut by threshold. Wrapper methods train models on feature subsets and score each subset by cross-validation performance; recursive feature elimination (RFE) is the canonical example. Embedded methods bake selection into training: L1 regularization drives coefficients to zero, tree-based models split on features and track usage. Filter is fastest but ignores feature interactions. Wrapper catches interactions but re-trains repeatedly. Embedded is a middle ground but ties you to a specific model architecture. The trade-off axis is compute cost vs. selection quality vs. model independence.

---

## Build It

Code implementing all three families on a synthetic dataset with known informative features. Generate 20 features where only 5 correlate with the target. Run correlation filtering, mutual information scoring, RFE with a Random Forest, and L1-penalized logistic regression. Print selected features for each method and compare overlap. Observable output: which features each method kept, training time, and cross-validation score.

**Exercise hooks:**
- *Easy:* Modify the correlation threshold and observe how the selected feature set changes
- *Medium:* Add two collinear features to the dataset and detect which method handles redundancy best
- *Hard:* Build a selector that combines filter pre-screening with RFE, keeping the speed of filter and the interaction awareness of wrapper

---

## Use It

GTM redirect: **Zone 2 — Scoring & Qualification**. When you enrich accounts in Clay or a similar tool, you accumulate dozens of signals — employee count, funding round, tech stack keywords, intent topics, engagement metrics. Not all of them predict conversion. Feature selection applied to historical win/loss data tells you which enrichment fields deserve weight in your scoring model and which are noise. This is the mechanism behind "which signals actually matter for ICP matching" — it's not guesswork, it's variance explained.

---

## Ship It

Feature selection is not a one-time operation. Input distributions shift: a signal that predicted conversion in Q2 may be noise by Q4. Ship a selection pipeline that logs which features were selected, at what threshold, with what performance metric. Add a scheduled re-evaluation: re-run selection monthly and diff against the previous set. If the selected feature set changes by more than 20%, flag for manual review. Store the selector object (not just the feature list) so you can reproduce the decision under the same parameters.

---

## Extend It

- **Permutation importance**: model-agnostic method that measures performance drop when a feature's values are shuffled; works on any fitted model
- **SHAP values**: decompose individual predictions into per-feature contributions; global aggregation gives feature importance with theoretical guarantees
- **Boruta**: wrapper method that compares real features against shadow copies (permuted duplicates) to decide which features beat random noise
- **Feature selection for time series**: lagged features introduce autocorrelation; standard selection methods assume independence and can mislead
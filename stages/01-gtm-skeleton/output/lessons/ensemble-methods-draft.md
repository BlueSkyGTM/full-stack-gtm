# Ensemble Methods

## Beat 1: Hook

Multiple models making independent errors can outperform any single model when their predictions are aggregated. Ensemble methods exploit this statistical fact — errors cancel, signal accumulates.

## Beat 2: Concept

Cover the three ensemble families and when each applies. Bagging (parallel bootstrap samples → variance reduction, e.g. Random Forest). Boosting (sequential error-weighted training → bias reduction, e.g. AdaBoost, Gradient Boosting, XGBoost). Stacking (heterogeneous base models + meta-learner). Majority voting and weighted averaging as the simplest ensembles. Bias-variance tradeoff as the deciding factor for which family to choose.

## Beat 3: Demo

Train a single decision tree, a bagged ensemble, and a boosted ensemble on the same noisy classification dataset. Print accuracy and variance across folds for each. Observable output: metrics table showing how ensemble depth changes error characteristics.

## Beat 4: Use It

GTM redirect: Clay's waterfall enrichment pattern is structurally a sequential ensemble — multiple data providers (ZoomInfo, Apollo, Hunter, etc.) queried in priority order, each filling gaps left by the previous. The mechanism mirrors boosting: each source corrects the failures of the prior pass. Reducing incomplete contact records follows the same error-reduction curve. Maps to **Zone 1 — Signal Capture** in the GTM topic map. [CITATION NEEDED — concept: Clay waterfall as boosting analogy]

## Beat 5: Ship It

**Easy:** Implement a majority-vote ensemble from three scikit-learn classifiers on a provided dataset; print accuracy vs. each individual model.

**Medium:** Build a stacking classifier with logistic regression as the meta-learner over a decision tree, SVM, and KNN base layer. Output the cross-val scores for each base model and the stacked ensemble side by side.

**Hard:** Simulate a GTM enrichment waterfall: given a list of 100 prospects with missing fields, query three mock provider functions (each with different coverage and accuracy), aggregate results, and print the before/after completion rate and estimated accuracy.

## Beat 6: Evaluate

Questions target the mechanism, not trivia:
- Why does bagging reduce variance but not bias?
- Given a dataset where single models underfit, which ensemble family addresses this and why?
- In a stacking architecture, what happens if all base learners make the same type of error?
- Map the components of a boosting ensemble (base learner, error weights, sequential correction) to the components of a multi-source enrichment waterfall.

---

**Learning Objectives (for `docs/en.md`):**
1. Implement bagging, boosting, and stacking ensembles using scikit-learn.
2. Compare the error-reduction mechanisms of bagging versus boosting on the same dataset.
3. Diagnose whether a given problem exhibits high bias or high variance to select the correct ensemble family.
4. Configure a stacking meta-learner over heterogeneous base models and evaluate its performance against individual models.
5. Map the boosting pattern (sequential error correction) to multi-source data enrichment workflows in GTM systems.
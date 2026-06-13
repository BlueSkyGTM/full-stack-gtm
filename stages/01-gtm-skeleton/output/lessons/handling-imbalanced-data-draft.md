# Handling Imbalanced Data

## Hook

You built a lead-scoring model. It achieves 96% accuracy. It predicts "no" on every single lead. Accuracy lied to you — welcome to class imbalance.

## Learn It

Mechanism walk-through: why imbalance breaks models (loss function dominated by majority class), the three correction strategies (resample, reweight, rethreshold), and when each applies. Covers SMOTE's interpolation algorithm, class_weight's loss penalty mechanism, and why decision thresholds are not fixed at 0.5.

## Use It

GTM redirect: lead scoring and conversion prediction, where closed-won deals are 2–5% of records — the textbook minority-class problem. Map each correction strategy to CRM pipeline data. Introduce `imbalanced-learn` as the library that implements SMOTE, Tomek links, and EasyEnsemble. Mechanism first: SMOTE interpolates synthetic minority samples along lines connecting nearest neighbors; `imbalanced-learn` packages this in a `fit_resample` API compatible with scikit-learn pipelines.

**Exercise hooks:**
- *Easy:* Load a 95/5 imbalanced dataset, print class distribution before and after `RandomOverSampler`.
- *Medium:* Build a pipeline: SMOTE → StandardScaler → LogisticRegression. Compare F1 scores with and without resampling.
- *Hard:* Implement threshold tuning on a trained classifier: sweep 0.1–0.9, plot precision-recall tradeoff, select threshold that maximizes F1. Print the chosen threshold and corresponding metrics.

## Build It

Full working script: generate an imbalanced binary classification dataset (97/3 split), train two models (naive vs class_weight='balanced'), evaluate with classification report + confusion matrix + PR-AUC. Observable output: printed metrics tables and a text-based confusion matrix. No scaffolding — every line runs.

## Ship It

GTM redirect: when your lead-score model ships to the CRM, the business does not care about accuracy — it cares about pipeline generated from correctly identified high-intent accounts. Threshold selection is a business decision (precision vs recall tradeoff), not a data science decision. Document how to log the chosen threshold and monitor class distribution drift in production.

## Evaluate

Assessment targets: explain why accuracy is misleading for imbalanced problems, implement SMOTE resampling in a pipeline, configure class_weight to penalize minority class misclassification, interpret a precision-recall curve, and justify threshold selection given a business constraint on minimum precision.

**Quiz hooks (not full items):**
- Given a 98/2 dataset and a model that always predicts majority class, calculate accuracy — explain why this is unacceptable.
- Identify which resampling strategy is appropriate when the minority class has fewer than 50 samples (hint: SMOTE's kNN requirement breaks down).
- Given a PR curve, select the threshold that achieves ≥80% precision while maximizing recall.
- Compare the mechanism of class_weight='balanced' vs SMOTE: which modifies the data, which modifies the loss?

---

**Learning Objectives:**
1. Detect and quantify class imbalance in a labeled dataset using distribution ratios and baseline accuracy comparison.
2. Implement SMOTE oversampling and random undersampling using `imbalanced-learn` within a scikit-learn pipeline.
3. Configure `class_weight` parameters to impose asymmetric loss penalties on minority class misclassification.
4. Compare model performance using precision-recall curves and PR-AUC instead of accuracy and ROC-AUC.
5. Evaluate and select decision thresholds based on business constraints on precision or recall minimums.

**GTM Cluster Reference:** Zone 3 — Pipeline Scoring & Conversion Prediction [CITATION NEEDED — concept: lead scoring model evaluation metrics in GTM context]
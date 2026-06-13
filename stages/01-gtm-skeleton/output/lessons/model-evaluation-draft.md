# Model Evaluation

## Hook

You deployed a lead-scoring model. Precision is 92%. Your reps say the leads are garbage. The metric lied — or you measured the wrong thing. This lesson is about picking metrics that match the business constraint, not just the math.

## Concept

Cover the mechanism ladder: confusion matrix → derived metrics (precision, recall, F1) → threshold-dependent vs threshold-independent evaluation (ROC-AUC) → calibration. Then LLM-specific eval: reference-based (BLEU, ROUGE, BERTScore) vs reference-free (LLM-as-judge, rubric scoring). Emphasize the selection algorithm: "what is the cost of a false positive vs a false negative?" determines the metric, not the other way around.

## Demo

Build a confusion matrix from scratch on a binary classification output, compute precision/recall/F1, then sweep thresholds to plot ROC and locate the optimal operating point. Print everything to terminal. Second script: evaluate a set of LLM-generated emails against reference outputs using ROUGE-L, with observable scores.

## Use It

**GTM Cluster: Zone 1 — Targeting (Lead Scoring & ICP Matching)**

Lead scoring is a binary classifier: predict "will close" vs "won't close." The cost asymmetry is extreme — a false positive wastes rep time, a false negative loses a deal. Walk through mapping that asymmetry to metric selection (precision@k, recall at threshold) and how to evaluate a scoring model against actual pipeline outcomes. If the model is an LLM classifying intent, the same eval frame applies but the scoring rubric replaces the confusion matrix.

## Ship It

Production eval pipeline pattern: log predictions alongside outcomes, compute rolling metrics weekly, alert on drift. Cover the mechanism of a "eval harness" — a script that runs on a cadence, pulls labeled data, re-computes metrics, and writes results to a dashboard or Slack alert. Mention tools that implement this (Evidently, WhyLabs) but focus on the pattern.

## Debug It

Catalog the failure modes: class imbalance makes accuracy meaningless, threshold defaults (0.5) are almost always wrong, leaked labels inflate all metrics, and LLM-as-judge has sycophancy bias. For each, describe the symptom and the diagnostic check.

---

### Exercise Hooks

| Difficulty | Exercise |
|------------|----------|
| Easy | Compute precision, recall, F1 from a given confusion matrix. Print results. |
| Medium | Load predictions, sweep 50 thresholds, plot ROC curve data to terminal, report AUC and best threshold by F1. |
| Hard | Build an eval harness: take a predictions file and an outcomes file, compute all metrics, detect if precision dropped >5% week-over-week, and print a pass/fail report. |

---

### Learning Objectives

1. Compute precision, recall, and F1 from raw prediction counts given a confusion matrix
2. Sweep classification thresholds to find the operating point that optimizes a chosen metric
3. Detect overfitting by comparing training metrics to held-out validation metrics
4. Evaluate LLM text outputs against references using ROUGE-L with observable scores
5. Diagnose class imbalance as a root cause when accuracy conflicts with business outcomes
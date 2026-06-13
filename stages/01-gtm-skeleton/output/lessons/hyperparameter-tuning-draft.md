# Hyperparameter Tuning

## Learning Objectives
1. Compare grid search, random search, and Bayesian optimization as search strategies over hyperparameter space.
2. Implement k-fold cross-validation to evaluate hyperparameter configurations without leaking test data.
3. Configure a search space and tuning loop for a lead-scoring classification model.
4. Detect overfitting caused by tuning on the test set instead of held-out validation.
5. Evaluate whether a tuning run produced meaningful improvement over default parameters.

---

## Hook It
You shipped a lead-scoring model. It looked great in testing. In production, it's random. The defaults betrayed you — but so did your "tuning." This lesson covers why tuning breaks models when done wrong, and the search strategies that actually find better configurations.

## Define It
Covers the mechanism: hyperparameters are configuration set before training (learning rate, tree depth, regularization strength); parameters are learned from data. Defines the search problem (combinatorial explosion over a space), the evaluation mechanism (cross-validation, not test-set peeking), and the three dominant search algorithms: grid search (exhaustive), random search (sampled), and Bayesian optimization (modeled). States tradeoffs: compute budget vs. coverage vs. intelligence.

## Show It
Demonstrates all three search strategies on a synthetic classification dataset. Prints the best configuration, best score, and total evaluations for each method so the practitioner sees the efficiency difference. Code runs end-to-end with `scikit-learn` and prints observable output — no visualization, terminal only. Shows a naive "tune on test set" example alongside the correct CV approach to make the overfitting mechanism visible.

## Use It
Redirect: Zone 3, lead scoring and prioritization. The practitioner has a classification model predicting conversion from firmographic and behavioral signals. This section frames hyperparameter tuning as the mechanism to improve precision-at-k without inflating overall accuracy via test-set leakage. References the Clay waterfall indirectly: enrichment cascades produce the feature set; tuning ensures the downstream model actually uses those features well. Exercise hooks: Easy — run `GridSearchCV` on a pre-built lead-scoring pipeline; Medium — switch from grid to `RandomizedSearchCV` with a budget constraint and compare; Hard — implement a simple Bayesian optimization loop using `scikit-optimize` with a 5-fold CV objective.

## Ship It
Production considerations: logging every trial (configuration + metric), setting random seeds for reproducibility, avoiding data leakage during preprocessing inside CV folds (use `Pipeline`), and knowing when to stop (diminishing returns, budget exhausted). Includes a working script that logs tuning trials to a CSV file and prints a summary. Discusses the failure mode where tuning on historical data does not generalize to a new quarter's lead distribution.

## Drill It
Five drill prompts: (1) explain why tuning on the test set inflates scores, (2) predict which search strategy finds a better config in fewer trials given a sparsely important hyperparameter, (3) trace through a 3-fold CV split and compute the mean score, (4) identify the leakage in a pipeline that preprocesses before the CV split, (5) decide whether a 0.3% accuracy improvement justifies a 10x compute increase. Each prompt targets one objective. No quiz bank without `docs/en.md`.

---

**GTM Cluster Redirect:** Zone 3 — Scoring & Prioritization. Tuning is the mechanism that converts enriched signals into calibrated scores. Without it, the model runs factory defaults.
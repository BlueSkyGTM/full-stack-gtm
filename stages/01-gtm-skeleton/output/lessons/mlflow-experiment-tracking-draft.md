# MLflow Experiment Tracking

## Learning Objectives

1. Configure an MLflow tracking server and create experiments programmatically
2. Log parameters, metrics, and artifacts during model training runs
3. Compare multiple runs to identify optimal hyperparameter configurations
4. Retrieve and filter run data using the MLflow search API
5. Implement a multi-run training loop with automated logging

---

## Beat 1: Hook

**The problem:** You train a model Tuesday. By Thursday you've forgotten which learning rate, which feature set, and which random seed produced that 0.84 F1 score. Spreadsheets of run results break at scale and under collaboration.

**The redirect:** This is foundational for Zone 1 — any GTM workflow that produces a model (lead scoring, ICP classification, churn prediction) needs a reproducible record of what was tried and what worked.

---

## Beat 2: Concept

**Mechanism:** An experiment tracking system stores a write-once ledger of run metadata: parameters (inputs you set), metrics (outputs you measure), and artifacts (files you save — models, plots, data). Each run is immutable. Runs are grouped by experiment. Experiments are searchable.

**How MLflow implements this:** MLflow uses a client-server model. The tracking server (or local file store) receives logged data via REST API or local filesystem calls. Each run gets a unique ID, a timestamp, and belongs to exactly one experiment. The `mlflow.search_runs()` function queries the store with filter strings like `metrics.accuracy > 0.8`.

**Key data model:**
- Experiment → contains Runs
- Run → contains Parameters (key-value, string), Metrics (key-value, numeric, time-series), Tags (key-value, string), Artifacts (files)

**Why immutability matters:** Runs cannot be overwritten after logging. This prevents accidental erasure of results and enables reliable comparison across team members and time.

---

## Beat 3: Demo

**Working code:** A complete training loop that trains a simple classifier on synthetic data across three hyperparameter configurations, logs all parameters and metrics to MLflow, then queries the results to find the best run.

**Observable output:** Printed table of run IDs, parameters, and metrics sorted by accuracy, plus the artifact directory listing.

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("file:///tmp/mlruns")
experiment_id = mlflow.create_experiment("demo_classifier", artifact_location="/tmp/mlruns/artifacts")
mlflow.set_experiment("demo_classifier")

configs = [
    {"n_estimators": 10, "max_depth": 3},
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
]

for i, params in enumerate(configs):
    with mlflow.start_run(run_name=f"run_{i}"):
        mlflow.log_params(params)
        clf = RandomForestClassifier(random_state=42, **params)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(clf, "model")
        print(f"Run {i}: params={params}, accuracy={acc:.4f}, f1={f1:.4f}")

runs = mlflow.search_runs(experiment_ids=[experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=3)

print("\n--- Top Runs ---")
for _, row in runs.iterrows():
    print(f"Run: {row['run_id'][:8]}... | Acc: {row['metrics.accuracy']:.4f} | "
          f"n_est: {row['params.n_estimators']} | depth: {row['params.max_depth']}")
```

---

## Beat 4: Use It

**GTM Application:** In a lead scoring pipeline, you test multiple model configurations: different feature sets (firmographic only vs. firmographic + behavioral), different thresholds, different algorithms. MLflow tracks which configuration produced the best conversion prediction. When you deploy to production, you reference the specific run ID and can reproduce the exact model.

**Exercise hooks:**

- **Easy:** Log three runs with different feature subsets of a provided dataset. Query for the best F1 score.
- **Medium:** Build a function that accepts a parameter grid dictionary, runs all combinations, logs each as a separate MLflow run, and returns the run ID with the highest metric.
- **Hard:** Implement a "champion-challenger" pattern: load the current champion model from a logged artifact, compare it against a new challenger on the same test set, and promote the winner by logging a "champion" tag.

---

## Beat 5: Ship It

**Integration scenario:** Your training script runs in a CI/CD pipeline or a scheduled job. You need the tracking server to persist results across runs and be queryable by the team.

**Exercise hooks:**

- **Easy:** Configure MLflow to log to a shared file path (`file:///shared/mlruns`) and verify that a second script can read runs logged by the first.
- **Medium:** Write a script that loads the best model from an experiment by metric, runs inference on new data, and saves predictions to a CSV — end to end, no manual step.
- **Hard:** Set up a remote tracking server (using `mlflow server`), configure authentication, and log a run from a separate process. Document the server start command, the tracking URI, and how to verify the run landed.

---

## Beat 6: Review

**Key takeaways:**

1. MLflow's data model is: Experiment → Run → {Parameters, Metrics, Tags, Artifacts}. Runs are immutable.
2. Parameters are inputs (strings). Metrics are outputs (numbers). Artifacts are files.
3. `mlflow.search_runs()` with filter strings and `order_by` replaces manual comparison spreadsheets.
4. The tracking URI determines where data lands — local filesystem for development, remote server for teams.
5. Every production model should trace back to a specific run ID. No orphan models.

**GTM redirect:** This is foundational for Zone 1. Any GTM team shipping ML models — lead scoring, ICP classification, intent prediction — needs experiment tracking to know what's deployed and why it was chosen over alternatives.

---

**Citation status:** [CITATION NEEDED — concept: MLflow integration patterns in GTM/RevOps workflows for lead scoring model management]
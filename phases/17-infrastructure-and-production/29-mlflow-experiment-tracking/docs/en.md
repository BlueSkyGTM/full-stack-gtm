# MLflow Experiment Tracking

> You ran the same training script twelve times with different hyperparameters. Which run had the best validation loss? What were the exact parameters? Where is the model checkpoint? If you cannot answer these questions in under thirty seconds, you do not have experiment tracking — you have a folder of unnamed `.pt` files and a memory problem. MLflow solves this by making every training run a first-class artifact with a timestamp, parameters, metrics, and model registry entry.

**Type:** Build
**Languages:** Python (mlflow, scikit-learn, numpy)
**Prerequisites:** Phase 01 (Python fundamentals), Phase 02 (Linear algebra and optimization)
**Time:** ~75 minutes

## Learning Objectives

- Explain the four MLflow components (Tracking, Models, Model Registry, Projects) and when each applies.
- Instrument a training loop with `mlflow.log_param`, `mlflow.log_metric`, and `mlflow.log_artifact`.
- Use the MLflow UI to compare runs, inspect metrics over time, and locate the best checkpoint.
- Register a model in the Model Registry and transition it from Staging to Production.

## The Problem

Machine learning is empirical. You change the learning rate, retrain, check validation accuracy, change it again. After ten iterations you have forgotten which configuration produced the model you are currently using in staging. You regenerate results and they differ by 0.3% because you forgot to set a random seed. Your colleague wants to reproduce your best run and cannot because your notes say "lr=0.001 worked better" without specifying which dataset version or which preprocessing flags were active.

This is not a discipline problem. It is an infrastructure problem. Experiment tracking makes the state of every run explicit and queryable.

## The Concept

### The four MLflow components

**MLflow Tracking** is the core. Every training run writes to a *tracking server* — a local SQLite database by default, a remote PostgreSQL or S3-backed store in production. A run captures:
- **Parameters**: scalar configuration values set before training (`lr=0.001`, `batch_size=32`)
- **Metrics**: scalar values logged during training, keyed and timestamped (`train_loss`, `val_accuracy`)
- **Artifacts**: files logged after training (model checkpoints, plots, confusion matrices)
- **Tags**: free-form key-value strings for metadata (`git_commit`, `dataset_version`)

**MLflow Models** defines a standard packaging format. A logged model includes the model object, a `conda.yaml` or `requirements.txt`, and a `MLmodel` manifest. This lets any MLflow-aware serving tool load the model without knowing whether it was Scikit-learn, PyTorch, or a custom class.

**MLflow Model Registry** is a versioned catalog. You promote models through stages: `None → Staging → Production → Archived`. Each transition is auditable. The registry answers "what is the current production model and where did it come from?"

**MLflow Projects** packages code for reproducible remote execution. Less commonly used in 2026 — container-based pipelines (Docker, Ray) have largely replaced it for distributed work.

### Run lifecycle

```
mlflow.set_experiment("my-experiment")
with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    for epoch in range(epochs):
        loss = train_one_epoch(...)
        mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.sklearn.log_model(model, "model")
```

`mlflow.start_run()` creates a run, assigns it a UUID, and opens a context. Everything logged inside the context belongs to that run. When the context exits (or if the process crashes), the run is marked finished or failed respectively.

### Auto-logging

MLflow provides `mlflow.sklearn.autolog()`, `mlflow.pytorch.autolog()`, etc. Call once before training and MLflow instruments the library's standard callbacks automatically — no manual `log_metric` calls needed. Auto-logging captures what it can; manual logging lets you add domain-specific metrics the library does not know about.

### Comparing runs

The MLflow UI (run `mlflow ui`) shows all runs in a table. You can:
- Sort by any metric to find the best run
- Plot metric curves side by side across multiple runs
- Diff the parameters between two runs to see what changed

The API equivalent: `mlflow.search_runs(experiment_names=["my-experiment"])` returns a Pandas DataFrame of all runs, sortable and filterable.

### Model Registry workflow

```python
# After logging the model inside a run:
run_id = mlflow.active_run().info.run_id
model_uri = f"runs:/{run_id}/model"
mlflow.register_model(model_uri, "MyClassifier")

# Transition to production (programmatic):
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="MyClassifier", version=1, stage="Production"
)
```

Production serving tools query `models:/MyClassifier/Production` and always get the current production version. Promoting a new version is a registry operation, not a code change.

### Tracking server backends

| Backend | When to use |
|---------|-------------|
| Local `mlruns/` (default) | Solo dev, laptop experiments |
| Local SQLite | Team on shared filesystem |
| PostgreSQL + S3 artifacts | Team on shared cloud infra |
| Databricks MLflow (hosted) | Enterprise, managed |

The code does not change between backends — only the `MLFLOW_TRACKING_URI` environment variable.

## Build It

Build a training loop that logs everything to MLflow, then uses the API to find the best run and register that model.

The experiment: train a Logistic Regression classifier on the Iris dataset with different regularization strengths `C`. Log `C` as a parameter, `val_accuracy` as a metric per fold, and the trained model as an artifact. After all runs complete, use `mlflow.search_runs` to find the best `C` and register that model.

See `code/main.py` for the full implementation.

Running `python code/main.py` will:
1. Start a local MLflow tracking server (writes to `./mlruns/`)
2. Run three training experiments with `C = [0.01, 0.1, 1.0]`
3. Log parameters, cross-validation accuracy per fold, and the final model
4. Print the best run ID and best accuracy
5. Register the best model as `IrisClassifier` version 1

After running, open `mlflow ui` in the same directory and navigate to `http://localhost:5000` to see all runs in the UI.

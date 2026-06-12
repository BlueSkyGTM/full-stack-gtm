# Skill: MLflow Experiment Tracking

## What you built

A reproducible experiment sweep: three Logistic Regression models trained with different regularization strengths, every run logged to MLflow with parameters, per-fold metrics, and model artifacts. The best run promoted to Production in the Model Registry.

## Reuse pattern

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("mlruns")          # or MLFLOW_TRACKING_URI env var
mlflow.set_experiment("your-experiment")

with mlflow.start_run(run_name="descriptive-name"):
    mlflow.log_param("key", value)
    mlflow.log_metric("val_loss", loss, step=epoch)
    mlflow.sklearn.log_model(model, "model")

# Find best run
runs = mlflow.search_runs(
    experiment_names=["your-experiment"],
    order_by=["metrics.val_loss ASC"],
)
best_run_id = runs.iloc[0]["run_id"]

# Register
mlflow.register_model(f"runs:/{best_run_id}/model", "ModelName")
```

## Next steps

- Add `mlflow.sklearn.autolog()` before `fit()` to capture parameters automatically.
- Connect to a remote tracking server: `export MLFLOW_TRACKING_URI=http://tracking-server:5000`
- Integrate with DVC (Lesson 32) for data versioning alongside experiment tracking.

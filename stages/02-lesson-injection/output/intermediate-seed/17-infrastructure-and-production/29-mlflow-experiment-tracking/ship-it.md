## Ship It

Moving from local file storage to a shared tracking server is the step that makes experiment tracking useful for teams. A local `file:///tmp/mlruns` store serves one person on one machine. A tracking server on a shared host lets every team member log runs to the same store, compare results, and retrieve each other's models.

For GTM teams operating enrichment waterfalls and scoring models in production, this shared store is the backbone of model lifecycle management. Zone 17 frames this as "versioning your enrichment waterfalls, detecting when your scoring model drifts." The practical application: when your lead-scoring model's precision drops three months after deployment — because the market shifted, your ICP evolved, or your enrichment data provider changed their schema — the MLflow experiment history tells you exactly what the model was trained on, when it was trained, and what performance baseline it should be held against. You retrain with new data, log a new run to the same experiment, and the before/after comparison is automatic. Without this, scoring drift is invisible until someone notices the sales team complaining about lead quality.

The following script sets up logging to a remote tracking server (simulated here with a local SQLite-backed server for reproducibility) and demonstrates tag-based filtering — the pattern for distinguishing production runs from experimental ones.

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("sqlite:////tmp/mlflow.db")

experiment_name = "lead_score_prod"
try:
    mlflow.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    pass
mlflow.set_experiment(experiment_name)

X, y = make_classification(n_samples=1000, n_features=15, n_informative=8, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

configs = [
    {"C": 0.01, "stage": "experimental"},
    {"C": 0.1, "stage": "experimental"},
    {"C": 1.0, "stage": "candidate"},
]

for config in configs:
    with mlflow.start_run() as run:
        model = LogisticRegression(C=config["C"], max_iter=500, random_state=7)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("C", config["C"])
        mlflow.log_param("max_iter", 500)
        mlflow.log_param("features", "firmographic+technographic")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.set_tag("stage", config["stage"])
        mlflow.set_tag("model_type", "logistic_regression")
        mlflow.set_tag("dataset_version", "2024-01-15")

        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"C={config['C']:<5} stage={config['stage']:<13} acc={acc:.4f} f1={f1:.4f}")

print()

candidates = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string="tags.stage = 'candidate'",
    order_by=["metrics.f1_score DESC"]
)

print(f"Candidate runs (tagged stage='candidate'): {len(candidates)}")
if len(candidates) > 0:
    best = candidates.iloc[0]
    print(f"  Best candidate run_id: {best['run_id']}")
    print(f"  F1: {best['metrics.f1_score']:.4f}")
    print(f"  Model URI: runs:/{best['run_id']}/model")
    print("  -> Promote to production via mlflow.register_model()")
```

To run a shared tracking server for a team: `mlflow server --backend-store-uri postgresql://user:pass@db/mlflow --default-artifact-root s3://your-bucket/mlruns --host 0.0.0.0 --port 5000`. Team members set `mlflow.set_tracking_uri("http://your-server:5000")` in their scripts. The backend store (PostgreSQL) holds run metadata; the artifact root (S3) holds model files and other artifacts. This separation matters at scale — metadata needs fast queries (SQL), artifacts need bulk storage (object store).

To promote a model to production, use the Model Registry: `mlflow.register_model("runs:/<run_id>/model", "lead_score_model")` creates a registry entry with versioning. You then transition versions between stages (Staging → Production) via `client.transition_model_version_stage("lead_score_model", version=3, stage="Production")`. The registry maintains a version history and stage transitions, so you always know which model version is live and can roll back if performance degrades.
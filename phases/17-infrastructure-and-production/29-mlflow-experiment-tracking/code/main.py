"""
MLflow experiment tracking — Phase 17, Lesson 29.

Trains a Logistic Regression classifier on Iris with three values of C,
logging each run to MLflow. After all runs, finds the best run by
val_accuracy and registers that model in the Model Registry.

Run:
    pip install mlflow scikit-learn numpy
    python code/main.py
    mlflow ui   # then open http://localhost:5000
"""

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


EXPERIMENT_NAME = "iris-regularization-sweep"
REGULARIZATION_VALUES = [0.01, 0.1, 1.0]
CV_FOLDS = 5
RANDOM_STATE = 42


def run_experiment(C: float, X, y) -> str:
    """Train with one regularization strength, log everything, return run_id."""
    with mlflow.start_run(run_name=f"C={C}") as run:
        mlflow.log_param("C", C)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("max_iter", 1000)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, solver="lbfgs",
                                       max_iter=1000, random_state=RANDOM_STATE)),
        ])

        # Cross-validate and log per-fold accuracy
        kf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pipeline.fit(X_train, y_train)
            acc = pipeline.score(X_val, y_val)
            fold_scores.append(acc)
            mlflow.log_metric("fold_accuracy", acc, step=fold)

        mean_acc = float(np.mean(fold_scores))
        std_acc = float(np.std(fold_scores))
        mlflow.log_metric("val_accuracy", mean_acc)
        mlflow.log_metric("val_accuracy_std", std_acc)

        # Re-fit on full dataset and log the model
        pipeline.fit(X, y)
        mlflow.sklearn.log_model(pipeline, artifact_path="model",
                                 registered_model_name=None)

        print(f"  C={C:5.2f}  val_accuracy={mean_acc:.4f} ± {std_acc:.4f}  run_id={run.info.run_id[:8]}")
        return run.info.run_id


def find_best_run(experiment_name: str) -> tuple[str, float]:
    """Return (run_id, best_val_accuracy) for the experiment."""
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.val_accuracy DESC"],
    )
    if runs.empty:
        raise RuntimeError("No runs found")
    best = runs.iloc[0]
    return best["run_id"], best["metrics.val_accuracy"]


def register_best_model(run_id: str, model_name: str) -> None:
    """Register the model from run_id in the Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    print(f"  Registered {model_name} version {mv.version} from run {run_id[:8]}")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
    )
    print(f"  Transitioned {model_name} v{mv.version} -> Production")


def main() -> None:
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_iris()
    X, y = data.data, data.target

    print(f"Running {len(REGULARIZATION_VALUES)} experiments on Iris ({len(X)} samples):")
    for C in REGULARIZATION_VALUES:
        run_experiment(C, X, y)

    print("\nFinding best run...")
    best_run_id, best_acc = find_best_run(EXPERIMENT_NAME)
    print(f"  Best run: {best_run_id[:8]}  val_accuracy={best_acc:.4f}")

    print("\nRegistering best model...")
    register_best_model(best_run_id, "IrisClassifier")

    print("\nDone. Run `mlflow ui` to inspect runs at http://localhost:5000")


if __name__ == "__main__":
    main()

"""Tests for MLflow experiment tracking lesson."""

import os
import tempfile
import numpy as np
import pytest
from sklearn.datasets import load_iris

import mlflow
import mlflow.sklearn


@pytest.fixture(autouse=True)
def isolated_mlflow(tmp_path):
    """Run each test against a fresh local tracking store."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("test-experiment")
    yield
    mlflow.end_run()


def test_log_param_and_metric():
    with mlflow.start_run():
        mlflow.log_param("C", 0.1)
        mlflow.log_metric("val_accuracy", 0.95)
    run = mlflow.last_active_run()
    assert run.data.params["C"] == "0.1"
    assert run.data.metrics["val_accuracy"] == pytest.approx(0.95)


def test_log_metric_per_step():
    with mlflow.start_run():
        for step, loss in enumerate([1.0, 0.8, 0.6]):
            mlflow.log_metric("loss", loss, step=step)
    run = mlflow.last_active_run()
    assert "loss" in run.data.metrics
    # MLflow stores only the last value for a metric key
    assert run.data.metrics["loss"] == pytest.approx(0.6)


def test_sklearn_model_logging():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_iris()
    X, y = data.data[:30], data.target[:30]
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=500))])
    pipeline.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id

    loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    preds = loaded.predict(X)
    assert preds.shape == (30,)


def test_search_runs_finds_best():
    experiment_name = "search-test"
    mlflow.set_experiment(experiment_name)

    for acc in [0.80, 0.95, 0.87]:
        with mlflow.start_run():
            mlflow.log_metric("val_accuracy", acc)

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.val_accuracy DESC"],
    )
    assert not runs.empty
    assert runs.iloc[0]["metrics.val_accuracy"] == pytest.approx(0.95)


def test_run_produces_required_keys():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    data = load_iris()
    X, y = data.data, data.target
    C = 0.1

    with mlflow.start_run() as run:
        mlflow.log_param("C", C)
        mlflow.log_param("cv_folds", 3)
        pipeline = Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(C=C, max_iter=500))])
        scores = cross_val_score(pipeline, X, y, cv=3)
        mlflow.log_metric("val_accuracy", float(np.mean(scores)))

    r = mlflow.get_run(run.info.run_id)
    assert "C" in r.data.params
    assert "cv_folds" in r.data.params
    assert "val_accuracy" in r.data.metrics

"""Tests for the containerized FastAPI inference server — no Docker needed."""
import pickle
import sys
from pathlib import Path
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure code/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    """Train and save a model to a temp directory."""
    tmp = tmp_path_factory.mktemp("model")
    data = load_iris()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipeline.fit(data.data, data.target)
    model_path = tmp / "model.pkl"
    model_path.write_bytes(pickle.dumps(pipeline))
    return model_path


@pytest.fixture
def client(saved_model, monkeypatch):
    """Patch MODEL_PATH and return a TestClient for the app."""
    from fastapi.testclient import TestClient
    import main as app_module

    monkeypatch.setattr(app_module, "MODEL_PATH", saved_model)

    # Reset and reload the model synchronously for tests
    data = load_iris()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
    ])
    pipeline.fit(data.data, data.target)
    monkeypatch.setattr(app_module, "_model", pipeline)

    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        yield c


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_setosa(client):
    resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "setosa"
    assert body["class_index"] == 0
    assert len(body["probabilities"]) == 3
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-6


def test_predict_virginica(client):
    resp = client.post("/predict", json={"features": [6.3, 3.3, 6.0, 2.5]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "virginica"
    assert body["class_index"] == 2


def test_predict_wrong_feature_count(client):
    resp = client.post("/predict", json={"features": [5.1, 3.5]})
    assert resp.status_code == 422   # Pydantic validation error


def test_probabilities_sum_to_one(client):
    features_list = [
        [5.1, 3.5, 1.4, 0.2],
        [6.7, 3.0, 5.2, 2.3],
        [5.9, 3.0, 4.2, 1.5],
    ]
    for features in features_list:
        resp = client.post("/predict", json={"features": features})
        assert resp.status_code == 200
        total = sum(resp.json()["probabilities"])
        assert abs(total - 1.0) < 1e-5

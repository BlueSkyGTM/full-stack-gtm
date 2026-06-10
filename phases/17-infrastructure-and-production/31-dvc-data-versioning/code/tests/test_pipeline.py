"""Tests for DVC pipeline stage functions — no DVC CLI required."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from main import generate_dataset, preprocess, train


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "raw" / "dataset.csv"
    generate_dataset(path, n_samples=200)
    return path


def test_generate_dataset_shape(tmp_path):
    path = tmp_path / "dataset.csv"
    generate_dataset(path, n_samples=100)
    df = pd.read_csv(path)
    assert df.shape == (100, 11)          # 10 features + label
    assert set(df["label"].unique()) == {0, 1}


def test_preprocess_split_sizes(raw_csv, tmp_path):
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    preprocess(raw_csv, train_out, test_out)

    train_df = pd.read_parquet(train_out)
    test_df = pd.read_parquet(test_out)
    total = len(train_df) + len(test_df)
    assert total == 200
    assert 0.15 < len(test_df) / total < 0.25   # ~20% test split


def test_preprocess_preserves_label(raw_csv, tmp_path):
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    preprocess(raw_csv, train_out, test_out)
    for path in (train_out, test_out):
        df = pd.read_parquet(path)
        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})


def test_train_writes_metrics(raw_csv, tmp_path):
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    preprocess(raw_csv, train_out, test_out)

    model_out = tmp_path / "classifier.pkl"
    metrics_out = tmp_path / "scores.json"
    train(train_out, test_out, model_out, metrics_out)

    metrics = json.loads(metrics_out.read_text())
    assert "train_accuracy" in metrics
    assert "test_accuracy" in metrics
    assert 0.5 < metrics["test_accuracy"] <= 1.0


def test_train_model_is_loadable_and_predicts(raw_csv, tmp_path):
    train_out = tmp_path / "train.parquet"
    test_out = tmp_path / "test.parquet"
    preprocess(raw_csv, train_out, test_out)

    model_out = tmp_path / "classifier.pkl"
    metrics_out = tmp_path / "scores.json"
    train(train_out, test_out, model_out, metrics_out)

    scaler, clf = pickle.loads(model_out.read_bytes())
    X = pd.read_parquet(test_out).drop("label", axis=1).values
    preds = clf.predict(scaler.transform(X))
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})

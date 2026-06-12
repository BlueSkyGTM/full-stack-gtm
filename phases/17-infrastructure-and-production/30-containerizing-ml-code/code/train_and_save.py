"""Train and save a model for the Docker inference server demo."""
import pickle
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def main():
    data = load_iris()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    pipeline.fit(data.data, data.target)
    acc = pipeline.score(data.data, data.target)
    Path("model.pkl").write_bytes(pickle.dumps(pipeline))
    print(f"Saved model.pkl  train_accuracy={acc:.4f}")

if __name__ == "__main__":
    main()

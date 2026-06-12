"""
Containerizing ML code — Phase 17, Lesson 30.

FastAPI inference server for the Iris classifier.
Demonstrates: health endpoint, predict endpoint, model loading at startup,
input validation with Pydantic, and structured JSON error responses.

Run locally (no Docker):
    pip install fastapi uvicorn scikit-learn pydantic
    python train_and_save.py        # creates model.pkl
    uvicorn main:app --port 8000

Run in Docker:
    python train_and_save.py
    docker build -t iris-server .
    docker run --rm -p 8000:8000 iris-server
"""

import pickle
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("model.pkl")
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    global _model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python train_and_save.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
    yield
    _model = None
    print("Model released")


app = FastAPI(
    title="Iris Classifier",
    description="Phase 17/30 — containerized Scikit-learn inference server",
    version="0.1.0",
    lifespan=lifespan,
)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[sepal_length, sepal_width, petal_length, petal_width] in cm",
        examples=[[5.1, 3.5, 1.4, 0.2]],
    )


class PredictResponse(BaseModel):
    predicted_class: str
    class_index: int
    probabilities: List[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness and readiness check."""
    return HealthResponse(status="ok", model_loaded=_model is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify one Iris sample."""
    if _model is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    import numpy as np
    X = np.array([request.features])
    class_idx = int(_model.predict(X)[0])
    probs = _model.predict_proba(X)[0].tolist()
    return PredictResponse(
        predicted_class=IRIS_CLASSES[class_idx],
        class_index=class_idx,
        probabilities=probs,
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)

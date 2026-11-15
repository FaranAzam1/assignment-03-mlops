"""Inference service deployed to Kubernetes. /health = liveness, /ready = readiness."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/models/model.joblib"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")

app = FastAPI(title="ML inference", version=MODEL_VERSION)
_bundle: dict | None = None


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=200)


@app.on_event("startup")
def load_model() -> None:
    global _bundle
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"model_version": MODEL_VERSION}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    x = np.array(req.features, dtype=float).reshape(1, -1)
    expected = _bundle["scaler"].n_features_in_
    if x.shape[1] != expected:
        raise HTTPException(status_code=422, detail=f"expected {expected} features, got {x.shape[1]}")
    xs = _bundle["scaler"].transform(x)
    model = _bundle["model"]
    return {"prediction": int(model.predict(xs)[0]), "probability": float(model.predict_proba(xs)[0, 1]),
            "model_version": MODEL_VERSION}

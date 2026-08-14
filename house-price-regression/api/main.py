"""
FastAPI service for the house-price regression model.

    uvicorn api.main:app --reload --port 8000
    open http://localhost:8000/docs

The model is loaded once at startup (not per-request) and reused across
requests. Feature engineering, imputation, encoding, and scaling all live
inside the saved pipeline (see src/features/build_features.py) -- this
endpoint never reimplements any of that logic, which is exactly what
prevents train/serve skew.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    HouseFeatures,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import resolve
from src.models.predict import load_production_model, predict_batch, predict_price

STATE: dict = {"model": None, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["model"] = load_production_model()
    except FileNotFoundError as e:
        STATE["load_error"] = str(e)
    yield
    STATE.clear()


app = FastAPI(
    title="House Price Regression API",
    description="Serves predictions from a Ridge-regularized linear regression model "
    "trained on the house-price-regression portfolio project.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _require_model():
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {STATE['load_error'] or 'unknown error'}. "
            "Run `python -m src.models.train` first.",
        )
    return STATE["model"]


@app.get("/", tags=["meta"])
def root():
    return {"message": "House Price Regression API", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok" if STATE["model"] is not None else "model_not_loaded",
                           model_loaded=STATE["model"] is not None)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["meta"])
def model_info():
    metrics_path = resolve("reports", "metrics.json")
    if not metrics_path.exists():
        raise HTTPException(status_code=503, detail="No metrics.json found. Run training first.")
    with open(metrics_path) as f:
        summary = json.load(f)
    best = next(m for m in summary["metrics"] if m["model"] == summary["best_model"])
    return ModelInfoResponse(
        best_model=summary["best_model"], test_r2=best["test_r2"], test_rmse=best["test_rmse"],
        test_mae=best["test_mae"], n_features_final=summary["n_features_final"],
        n_train=summary["n_train"], n_test=summary["n_test"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: HouseFeatures):
    model = _require_model()
    price = predict_price(features.model_dump(), model=model)
    return PredictionResponse(
        predicted_price=round(price, 2), predicted_price_formatted=f"${price:,.0f}",
        model_used="Ridge (1-SE)",
    )


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch_endpoint(request: BatchPredictionRequest):
    model = _require_model()
    rows = [h.model_dump() for h in request.houses]
    prices = predict_batch(rows, model=model)
    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(predicted_price=round(p, 2), predicted_price_formatted=f"${p:,.0f}",
                                model_used="Ridge (1-SE)")
            for p in prices
        ]
    )

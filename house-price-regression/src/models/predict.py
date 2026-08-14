"""
Single entry point for turning a raw feature dict into a price prediction.
Used by the FastAPI service, the Streamlit demo, and tests -- so there is
exactly one code path between "raw input" and "prediction" everywhere in the
project.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import resolve
from src.features.build_features import (
    RAW_BOOLEAN_FEATURES,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
)

MODEL_PATH = resolve("models", "production_model.joblib")


@lru_cache(maxsize=1)
def load_production_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m src.models.train` first."
        )
    return joblib.load(path)


def _rows_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    expected = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES + RAW_BOOLEAN_FEATURES
    X = pd.DataFrame([{col: r.get(col) for col in expected} for r in rows])
    # Omitted optional fields arrive as None -> force proper float dtype (not
    # 'object') so the imputer/scaler treat them as NaN, not as a literal.
    for col in RAW_NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def predict_price(raw_input: dict[str, Any], model=None) -> float:
    """raw_input keys must match the RAW feature schema (see
    build_features.RAW_*_FEATURES) -- feature engineering, imputation,
    encoding, and scaling all happen inside the loaded pipeline."""
    model = model or load_production_model()
    X = _rows_to_frame([raw_input])
    return float(model.predict(X)[0])


def predict_batch(rows: list[dict[str, Any]], model=None) -> list[float]:
    model = model or load_production_model()
    X = _rows_to_frame(rows)
    return [float(p) for p in model.predict(X)]

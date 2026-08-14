"""Pydantic schemas for the prediction API.

Fields mirror the raw dataset schema (src/features/build_features.py) minus
the noise_1/2/3 columns, which are training-time-only sanity-check features
with no business meaning -- a real API has no reason to expose them, and the
pipeline's imputer fills them in (harmlessly, since they carry no signal) if
omitted.

The four fields we deliberately made Optional (garage_area, school_rating,
total_bsmt_sf, crime_index) mirror the same four columns the training data
generator injected missing-at-random values into -- the API's contract is
intentionally consistent with what the model was actually trained to handle.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Neighborhood = Literal["Downtown", "Lakeside", "Hillcrest", "Riverside", "Suburbia", "Old Town"]
HouseStyle = Literal["2Story", "1Story", "1.5Story", "SplitLevel"]


class HouseFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lot_area": 8500, "gr_liv_area": 1950, "total_bsmt_sf": 1000,
                "garage_area": 480, "year_built": 2005, "year_remod": 2015,
                "overall_qual": 7, "overall_cond": 6, "full_bath": 2, "half_bath": 1,
                "bedroom_abvgr": 3, "totrms_abvgrd": 7, "fireplaces": 1,
                "distance_to_downtown_km": 5.2, "school_rating": 8, "crime_index": 22.0,
                "median_income_area": 78.0, "has_pool": False,
                "neighborhood": "Lakeside", "house_style": "2Story",
            }
        }
    )

    lot_area: float = Field(..., gt=0, le=30000, description="Lot size, sq ft")
    gr_liv_area: float = Field(..., gt=0, le=6000, description="Above-ground living area, sq ft")
    total_bsmt_sf: float | None = Field(None, ge=0, le=4000, description="Basement area, sq ft")
    garage_area: float | None = Field(None, ge=0, le=1500, description="Garage area, sq ft")
    year_built: int = Field(..., ge=1800, le=2026)
    year_remod: int = Field(..., ge=1800, le=2026, description="Most recent remodel year")
    overall_qual: int = Field(..., ge=1, le=10, description="Overall material/finish quality, 1-10")
    overall_cond: int = Field(..., ge=1, le=10, description="Overall condition, 1-10")
    full_bath: int = Field(..., ge=0, le=10)
    half_bath: int = Field(0, ge=0, le=10)
    bedroom_abvgr: int = Field(..., ge=0, le=15, description="Bedrooms above grade")
    totrms_abvgrd: int = Field(..., ge=1, le=25, description="Total rooms above grade")
    fireplaces: int = Field(0, ge=0, le=5)
    distance_to_downtown_km: float = Field(..., ge=0, le=100)
    school_rating: float | None = Field(None, ge=1, le=10)
    crime_index: float | None = Field(None, ge=0, le=100)
    median_income_area: float = Field(..., gt=0, le=500, description="Area median income, $1000s")
    has_pool: bool = False
    neighborhood: Neighborhood
    house_style: HouseStyle


class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_used: str


class BatchPredictionRequest(BaseModel):
    houses: list[HouseFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    best_model: str
    test_r2: float
    test_rmse: float
    test_mae: float
    n_features_final: int
    n_train: int
    n_test: int

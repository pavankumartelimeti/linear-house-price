"""
Feature engineering.

This is where multicollinearity enters the pipeline — ON PURPOSE. In a real
team, an engineer adds `age_of_house` because "age matters for price" and
`total_sf` because "let's combine living area + basement into one signal."
Both are reasonable engineering decisions in isolation. Together, they
duplicate information already present in `year_built`, `gr_liv_area`, and
`total_bsmt_sf`, and that's exactly the setup where OLS coefficients become
unstable and Ridge/Lasso earn their keep. See notebooks/01_full_analysis for
the VIF evidence.

`FeatureEngineer` is a proper sklearn Transformer so it can live INSIDE the
same Pipeline as preprocessing and the model. That matters for two reasons:
  1. No train/serve skew — the API calls the exact same code path as training.
  2. Correct cross-validation — because it's inside the Pipeline, it gets
     re-fit (well, it's stateless here, but any future *fitted* engineering
     step would be) independently per fold, avoiding leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Raw columns expected as input (what a caller / API request must provide)
RAW_NUMERIC_FEATURES = [
    "lot_area", "gr_liv_area", "total_bsmt_sf", "garage_area",
    "year_built", "year_remod", "overall_qual", "overall_cond",
    "full_bath", "half_bath", "bedroom_abvgr", "totrms_abvgrd", "fireplaces",
    "distance_to_downtown_km", "school_rating", "crime_index",
    "median_income_area", "noise_1", "noise_2", "noise_3",
]
RAW_CATEGORICAL_FEATURES = ["neighborhood", "house_style"]
RAW_BOOLEAN_FEATURES = ["has_pool"]

# Columns created by FeatureEngineer — deliberately redundant with the above
ENGINEERED_FEATURES = ["age_of_house", "total_sf", "qual_x_cond"]

ALL_NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives age_of_house, total_sf, and qual_x_cond from raw columns.

    Stateless (fit is a no-op) so it's safe to place inside a CV-wrapped
    Pipeline without any leakage risk itself — but it still must live inside
    the Pipeline (not applied globally before the split) so that everything
    downstream of it is properly re-fit per fold.
    """

    def __init__(self, current_year: int = 2026):
        self.current_year = current_year

    def fit(self, X: pd.DataFrame, y=None) -> FeatureEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["age_of_house"] = self.current_year - X["year_built"]
        X["total_sf"] = X["gr_liv_area"].fillna(0) + X["total_bsmt_sf"].fillna(0)
        X["qual_x_cond"] = X["overall_qual"] * X["overall_cond"]
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        base = list(input_features) if input_features is not None else []
        return np.array(base + ENGINEERED_FEATURES)


def build_preprocessor() -> ColumnTransformer:
    """Impute -> encode -> scale. Column selection happens on the OUTPUT of
    FeatureEngineer, so engineered columns are included automatically."""
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, ALL_NUMERIC_FEATURES),
            ("cat", categorical_pipe, RAW_CATEGORICAL_FEATURES),
            ("bool", "passthrough", RAW_BOOLEAN_FEATURES),
        ],
        verbose_feature_names_out=True,
    )
    return preprocessor


def get_feature_engineering_pipeline(current_year: int = 2026) -> Pipeline:
    """FeatureEngineer + preprocessor, WITHOUT a model — useful on its own
    for VIF analysis / EDA on the final design matrix."""
    return Pipeline(
        [
            ("feature_engineer", FeatureEngineer(current_year=current_year)),
            ("preprocessor", build_preprocessor()),
        ]
    )


def get_clean_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Human-readable feature names after the ColumnTransformer, e.g.
    'num__gr_liv_area' -> 'gr_liv_area', 'cat__neighborhood_Downtown' ->
    'neighborhood_Downtown'."""
    raw_names = fitted_preprocessor.get_feature_names_out()
    cleaned = []
    for name in raw_names:
        for prefix in ("num__", "cat__", "bool__"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        cleaned.append(name)
    return cleaned


def load_raw_features_and_target(df: pd.DataFrame, target_col: str = "sale_price"):
    """Split a raw dataframe into X (raw feature columns only) and y."""
    feature_cols = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES + RAW_BOOLEAN_FEATURES
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y

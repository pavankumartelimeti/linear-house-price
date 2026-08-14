import numpy as np
import pytest

from src.data.make_dataset import generate_raw_dataset
from src.features.build_features import (
    ENGINEERED_FEATURES,
    FeatureEngineer,
    get_clean_feature_names,
    get_feature_engineering_pipeline,
    load_raw_features_and_target,
)


@pytest.fixture(scope="module")
def raw_df():
    return generate_raw_dataset(n=400, seed=11, current_year=2026)


def test_feature_engineer_adds_expected_columns(raw_df):
    X, _ = load_raw_features_and_target(raw_df)
    out = FeatureEngineer(current_year=2026).fit_transform(X)
    for col in ENGINEERED_FEATURES:
        assert col in out.columns


def test_age_of_house_matches_formula(raw_df):
    X, _ = load_raw_features_and_target(raw_df)
    out = FeatureEngineer(current_year=2026).transform(X)
    assert (out["age_of_house"] == 2026 - X["year_built"]).all()


def test_pipeline_output_has_no_missing_values(raw_df):
    X, _ = load_raw_features_and_target(raw_df)
    pipe = get_feature_engineering_pipeline()
    Xt = pipe.fit_transform(X)
    assert not np.isnan(np.asarray(Xt)).any()


def test_pipeline_handles_unseen_categorical_gracefully(raw_df):
    X, _ = load_raw_features_and_target(raw_df)
    pipe = get_feature_engineering_pipeline()
    pipe.fit(X)
    X_new = X.iloc[:3].copy()
    X_new["neighborhood"] = "Neighborhood That Does Not Exist"
    # handle_unknown="ignore" on the encoder -- should not raise
    Xt = pipe.transform(X_new)
    assert Xt.shape[0] == 3


def test_feature_names_are_clean(raw_df):
    X, _ = load_raw_features_and_target(raw_df)
    pipe = get_feature_engineering_pipeline()
    pipe.fit(X)
    names = get_clean_feature_names(pipe.named_steps["preprocessor"])
    assert not any(n.startswith(("num__", "cat__")) for n in names)
    assert "gr_liv_area" in names
    assert "neighborhood_Downtown" in names

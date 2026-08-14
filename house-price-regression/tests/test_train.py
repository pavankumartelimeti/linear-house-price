import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from src.data.make_dataset import generate_raw_dataset
from src.features.build_features import load_raw_features_and_target
from src.models.train import (
    evaluate_on_test,
    get_inner_coefficients,
    make_production_pipeline,
    make_search_pipeline,
    select_alpha_1se,
)


def test_select_alpha_1se_picks_larger_alpha_within_one_se():
    # Constructed so alpha=1 is the min, but 5 and 10 are within 1 SE of it.
    results = pd.DataFrame(
        {
            "alpha": [0.1, 1, 5, 10, 50],
            "mean_rmse": [0.20, 0.10, 0.11, 0.115, 0.30],
            "se_rmse": [0.01, 0.01, 0.01, 0.01, 0.01],
        }
    )
    alpha_min, alpha_1se = select_alpha_1se(results)
    assert alpha_min == 1
    # threshold = 0.10 + 0.01 = 0.11 -> candidates are alpha in {1, 5} (0.115 > 0.11 excludes 10)
    assert alpha_1se == 5


def test_select_alpha_1se_never_returns_worse_than_min():
    results = pd.DataFrame({"alpha": [1, 10, 100], "mean_rmse": [0.5, 0.4, 0.3], "se_rmse": [0.05] * 3})
    alpha_min, alpha_1se = select_alpha_1se(results)
    assert alpha_min == 100  # min error is at the last (largest) alpha here
    assert alpha_1se == 100  # nothing larger to be "more conservative" with


@pytest.fixture(scope="module")
def small_raw_data():
    df = generate_raw_dataset(n=300, seed=5, current_year=2026)
    return load_raw_features_and_target(df)


def test_production_pipeline_fits_and_predicts_in_dollar_scale(small_raw_data):
    X, y = small_raw_data
    pipe = make_production_pipeline(Ridge(alpha=10))
    pipe.fit(X, y)
    preds = pipe.predict(X.iloc[:5])
    # predictions should be in raw dollar scale, same order of magnitude as y
    assert preds.min() > 1000
    assert preds.max() < 5_000_000


def test_evaluate_on_test_returns_expected_keys(small_raw_data):
    X, y = small_raw_data
    pipe = make_production_pipeline(Ridge(alpha=10))
    pipe.fit(X.iloc[:250], y.iloc[:250])
    m = evaluate_on_test(pipe, X.iloc[250:], y.iloc[250:])
    for key in ["test_rmse", "test_mae", "test_r2", "test_adj_r2", "test_mape_pct"]:
        assert key in m
        assert np.isfinite(m[key])


def test_get_inner_coefficients_unwraps_transformed_target_regressor(small_raw_data):
    X, y = small_raw_data
    pipe = make_production_pipeline(Ridge(alpha=10))
    pipe.fit(X, y)
    coef = get_inner_coefficients(pipe)
    assert isinstance(coef, pd.Series)
    assert len(coef) > 0
    assert "gr_liv_area" in coef.index


def test_search_pipeline_operates_in_log_space_directly(small_raw_data):
    # make_search_pipeline (used only during CV) should NOT wrap in
    # TransformedTargetRegressor -- caller is responsible for pre-logging y.
    X, y = small_raw_data
    pipe = make_search_pipeline(Ridge(alpha=10))
    pipe.fit(X, np.log1p(y))
    preds = pipe.predict(X.iloc[:5])
    assert preds.max() < 20  # log-price scale, not dollar scale

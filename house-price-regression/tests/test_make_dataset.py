import pandas as pd
import pytest

from src.data.make_dataset import generate_raw_dataset


@pytest.fixture(scope="module")
def raw_df():
    return generate_raw_dataset(n=500, seed=123, current_year=2026)


def test_shape_and_no_duplicate_ids(raw_df):
    assert len(raw_df) == 500
    assert raw_df["id"].is_unique


def test_expected_columns_present(raw_df):
    expected = {"sale_price", "neighborhood", "house_style", "gr_liv_area", "year_built",
                "noise_1", "noise_2", "noise_3"}
    assert expected.issubset(set(raw_df.columns))


def test_sale_price_always_positive(raw_df):
    assert (raw_df["sale_price"] > 0).all()


def test_missingness_only_in_expected_columns(raw_df):
    cols_with_na = set(raw_df.columns[raw_df.isna().any()])
    assert cols_with_na == {"garage_area", "school_rating", "total_bsmt_sf", "crime_index"}


def test_generation_is_reproducible_given_same_seed():
    df1 = generate_raw_dataset(n=200, seed=7, current_year=2026)
    df2 = generate_raw_dataset(n=200, seed=7, current_year=2026)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_give_different_data():
    df1 = generate_raw_dataset(n=200, seed=1, current_year=2026)
    df2 = generate_raw_dataset(n=200, seed=2, current_year=2026)
    assert not df1["sale_price"].equals(df2["sale_price"])


def test_noise_columns_are_uncorrelated_with_true_price(raw_df):
    # Not exactly 0 in a finite sample, but should be small.
    for col in ["noise_1", "noise_2", "noise_3"]:
        corr = raw_df[col].corr(raw_df["true_price_no_noise"])
        assert abs(corr) < 0.15

"""
Synthetic housing-listing dataset generator.

Why synthetic? Because we control the data-generating process (DGP), we know
the *true* coefficients — something no real-world dataset gives you. That
lets later analysis do things you normally can't: verify whether Lasso really
recovers the true zero coefficients, or whether Ridge really reduces
coefficient variance versus OLS, against ground truth rather than folklore.

The dataset simulates ~3,000 home sales in a fictitious mid-sized city. It
deliberately includes real-world-flavored mess:
  - a handful of features with ZERO true relationship to price (to verify
    Lasso actually zeroes them out)
  - missing-at-random values in a few columns (to justify imputation)
  - a small fraction of outlier sales (flips / distressed sales)
  - a right-skewed price distribution (to justify a log-target transform)

Note multicollinearity is NOT baked in here — it's introduced later, on
purpose, during feature engineering (see src/features/build_features.py).
That mirrors how it usually happens in real pipelines: nobody hand-codes
correlated raw inputs, engineers just derive redundant features without
noticing.

Run:
    python -m src.data.make_dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_config, resolve

NEIGHBORHOODS = {
    # name: (price premium $, sampling weight)
    "Downtown": (48_000, 0.16),
    "Lakeside": (65_000, 0.12),
    "Hillcrest": (32_000, 0.18),
    "Riverside": (21_000, 0.14),
    "Suburbia": (0, 0.28),
    "Old Town": (-14_000, 0.12),
}

HOUSE_STYLES = {
    "2Story": (12_000, 0.30),
    "1Story": (0, 0.35),
    "1.5Story": (4_000, 0.15),
    "SplitLevel": (-3_000, 0.20),
}

# Tuned empirically (see notebook Section 2) so the *true* linear model
# explains ~85-90% of price variance — realistic for a real-estate model,
# not suspiciously perfect. (Verified via cross-validated OLS, not just the
# nominal formula, since outlier injection also adds residual variance.)
NOISE_STD_MULTIPLIER = 0.22


def _weighted_choice(rng: np.random.Generator, mapping: dict, n: int) -> np.ndarray:
    keys = list(mapping.keys())
    weights = np.array([mapping[k][1] for k in keys])
    weights = weights / weights.sum()
    return rng.choice(keys, size=n, p=weights)


def generate_raw_dataset(n: int, seed: int, current_year: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    neighborhood = _weighted_choice(rng, NEIGHBORHOODS, n)
    house_style = _weighted_choice(rng, HOUSE_STYLES, n)

    lot_area = np.clip(rng.lognormal(mean=9.0, sigma=0.35, size=n), 1800, 30_000)
    gr_liv_area = np.clip(rng.normal(1850, 550, n), 500, 5200)
    total_bsmt_sf = np.clip(rng.normal(950, 420, n), 0, 3200)

    has_garage = rng.random(n) > 0.12  # ~12% of homes have no garage
    garage_area = np.where(has_garage, np.clip(rng.normal(460, 160, n), 120, 1200), 0.0)

    year_built = rng.integers(1900, current_year, n)
    year_remod = np.minimum(year_built + rng.integers(0, 41, n), current_year)

    overall_qual = rng.integers(1, 11, n)
    overall_cond = rng.integers(1, 11, n)

    full_bath = rng.integers(1, 4, n)
    half_bath = rng.integers(0, 3, n)
    bedroom_abvgr = rng.integers(2, 7, n)
    totrms_abvgrd = np.clip(bedroom_abvgr + full_bath + rng.integers(-1, 3, n), 3, 14)
    fireplaces = rng.integers(0, 3, n)

    distance_to_downtown_km = np.clip(rng.exponential(6, n) + 0.4, 0.3, 28)
    school_rating = rng.integers(1, 11, n).astype(float)
    crime_index = np.clip(rng.normal(35, 20, n), 0, 100)
    median_income_area = np.clip(rng.normal(68, 18, n), 22, 180)
    has_pool = rng.random(n) < 0.14

    # --- Pure noise: MUST have zero relationship with price. These exist so
    # we can later confirm Lasso actually drives their coefficients to 0. ---
    noise_1 = rng.normal(0, 1, n)
    noise_2 = rng.uniform(0, 1, n)
    noise_3 = rng.integers(0, 100, n).astype(float)

    df = pd.DataFrame(
        {
            "neighborhood": neighborhood,
            "house_style": house_style,
            "lot_area": lot_area.round(0),
            "gr_liv_area": gr_liv_area.round(0),
            "total_bsmt_sf": total_bsmt_sf.round(0),
            "garage_area": garage_area.round(0),
            "year_built": year_built,
            "year_remod": year_remod,
            "overall_qual": overall_qual,
            "overall_cond": overall_cond,
            "full_bath": full_bath,
            "half_bath": half_bath,
            "bedroom_abvgr": bedroom_abvgr,
            "totrms_abvgrd": totrms_abvgrd,
            "fireplaces": fireplaces,
            "distance_to_downtown_km": distance_to_downtown_km.round(2),
            "school_rating": school_rating,
            "crime_index": crime_index.round(1),
            "median_income_area": median_income_area.round(1),
            "has_pool": has_pool,
            "noise_1": noise_1.round(3),
            "noise_2": noise_2.round(3),
            "noise_3": noise_3,
        }
    )

    # ---------------- Ground-truth, mostly-linear price DGP ----------------
    neigh_premium = df["neighborhood"].map({k: v[0] for k, v in NEIGHBORHOODS.items()})
    style_premium = df["house_style"].map({k: v[0] for k, v in HOUSE_STYLES.items()})

    true_price = (
        42_000
        + 58 * df["gr_liv_area"]
        + 34 * df["total_bsmt_sf"]
        + 55 * df["garage_area"]
        + 7_800 * df["overall_qual"]
        + 2_600 * df["overall_cond"]
        + 1.1 * df["lot_area"]
        + neigh_premium
        + style_premium
        + 16_500 * df["has_pool"]
        + 1_450 * df["fireplaces"]
        + 2_100 * df["full_bath"]
        + 950 * df["half_bath"]
        - 780 * df["distance_to_downtown_km"]
        + 2_950 * df["school_rating"]
        - 410 * df["crime_index"]
        + 870 * df["median_income_area"]
        + 180 * df["overall_qual"] * df["fireplaces"]  # small realistic interaction
        # noise_1 / noise_2 / noise_3 deliberately contribute NOTHING
    )

    sigma = NOISE_STD_MULTIPLIER * true_price.std()
    price = true_price + rng.normal(0, sigma, n)

    # A few outlier sales: flips / distressed sales (mild, so they add
    # realistic texture and a robustness talking point without swamping the
    # overall signal)
    outlier_idx = rng.choice(n, size=int(0.010 * n), replace=False)
    price[outlier_idx] *= rng.uniform(1.25, 1.6, size=len(outlier_idx))
    remaining = np.setdiff1d(np.arange(n), outlier_idx)
    distressed_idx = rng.choice(remaining, size=int(0.008 * n), replace=False)
    price[distressed_idx] *= rng.uniform(0.65, 0.8, size=len(distressed_idx))

    df["sale_price"] = np.clip(price, 35_000, None).round(-2)
    df["true_price_no_noise"] = true_price.round(0)  # kept ONLY for validation notebooks

    # ---------------- Inject realistic missing-at-random values ------------
    def inject_missing(col: str, frac: float, mask: np.ndarray | None = None) -> None:
        pool = df.index if mask is None else df.index[mask]
        n_missing = int(frac * len(pool))
        miss_idx = rng.choice(pool, size=n_missing, replace=False)
        df.loc[miss_idx, col] = np.nan

    inject_missing("garage_area", 0.05, mask=(df["garage_area"].values > 0))
    inject_missing("school_rating", 0.06)
    inject_missing("total_bsmt_sf", 0.03)
    inject_missing("crime_index", 0.02)

    df.insert(0, "id", np.arange(1, n + 1))
    return df


def main() -> pd.DataFrame:
    cfg = get_config()
    df = generate_raw_dataset(
        n=cfg["data"]["n_samples"],
        seed=cfg["random_seed"],
        current_year=cfg["current_year"],
    )
    out_path = resolve(cfg["data"]["raw_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df):,} rows x {df.shape[1]} cols -> {out_path}")
    print("\nMissingness (top columns):")
    print(df.isna().mean().sort_values(ascending=False).head(6).round(4))
    print("\nsale_price summary:")
    print(df["sale_price"].describe().round(0))
    return df


if __name__ == "__main__":
    main()

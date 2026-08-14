"""
Interactive demo -- lets you tweak a house's features and see how the
prediction (and the underlying model's confidence in the inputs) responds.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.predict import load_production_model, predict_price
from src.models.train import get_inner_coefficients

st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="wide")


@st.cache_resource
def get_model():
    return load_production_model()


st.title("🏡 House Price Predictor")
st.caption(
    "Ridge regression (1-SE rule), trained on a synthetic-but-realistic listings dataset. "
    "See the full write-up in `README.md` and `notebooks/01_full_analysis.ipynb`."
)

try:
    model = get_model()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.models.train` first, then reload this page.")
    st.stop()

col_form, col_result = st.columns([3, 2], gap="large")

with col_form:
    st.subheader("Property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        neighborhood = st.selectbox("Neighborhood", ["Downtown", "Lakeside", "Hillcrest",
                                                       "Riverside", "Suburbia", "Old Town"], index=4)
        house_style = st.selectbox("House style", ["2Story", "1Story", "1.5Story", "SplitLevel"])
        year_built = st.slider("Year built", 1900, 2026, 2005)
        year_remod = st.slider("Year remodeled", year_built, 2026, max(year_built, 2015))
        overall_qual = st.slider("Overall quality (1-10)", 1, 10, 7)
        overall_cond = st.slider("Overall condition (1-10)", 1, 10, 6)
    with c2:
        gr_liv_area = st.slider("Living area (sq ft)", 500, 5200, 1950, step=50)
        total_bsmt_sf = st.slider("Basement area (sq ft)", 0, 3200, 1000, step=50)
        lot_area = st.slider("Lot area (sq ft)", 1800, 30000, 8500, step=100)
        garage_area = st.slider("Garage area (sq ft)", 0, 1200, 480, step=20)
        full_bath = st.slider("Full baths", 0, 4, 2)
        half_bath = st.slider("Half baths", 0, 3, 1)
    with c3:
        bedroom_abvgr = st.slider("Bedrooms", 0, 8, 3)
        totrms_abvgrd = st.slider("Total rooms", 1, 16, 7)
        fireplaces = st.slider("Fireplaces", 0, 3, 1)
        has_pool = st.checkbox("Has pool", value=False)
        distance_to_downtown_km = st.slider("Distance to downtown (km)", 0.0, 28.0, 5.2)
        school_rating = st.slider("School rating (1-10)", 1, 10, 8)
        crime_index = st.slider("Crime index (0=low, 100=high)", 0, 100, 22)
        median_income_area = st.slider("Area median income ($1000s)", 22, 180, 78)

payload = {
    "lot_area": lot_area, "gr_liv_area": gr_liv_area, "total_bsmt_sf": total_bsmt_sf,
    "garage_area": garage_area, "year_built": year_built, "year_remod": year_remod,
    "overall_qual": overall_qual, "overall_cond": overall_cond, "full_bath": full_bath,
    "half_bath": half_bath, "bedroom_abvgr": bedroom_abvgr, "totrms_abvgrd": totrms_abvgrd,
    "fireplaces": fireplaces, "distance_to_downtown_km": distance_to_downtown_km,
    "school_rating": school_rating, "crime_index": crime_index,
    "median_income_area": median_income_area, "has_pool": has_pool,
    "neighborhood": neighborhood, "house_style": house_style,
    "noise_1": 0.0, "noise_2": 0.0, "noise_3": 0.0,
}

with col_result:
    st.subheader("Prediction")
    price = predict_price(payload, model=model)
    st.metric("Estimated sale price", f"${price:,.0f}")

    st.divider()
    st.subheader("What's driving this model")
    coef = get_inner_coefficients(model).abs().sort_values(ascending=False).head(8)
    st.bar_chart(coef.rename("‖standardized coefficient‖"))
    st.caption(
        "Top 8 features by absolute standardized coefficient in the underlying "
        "Ridge model — larger bars move the prediction more per standard "
        "deviation of that feature."
    )

st.divider()
with st.expander("Why Ridge instead of plain Linear Regression? (the short version)"):
    st.markdown(
        "Two engineered features here — `age_of_house` and `total_sf` — are "
        "highly redundant with columns already in the model (`year_built`, and "
        "`gr_liv_area` + `total_bsmt_sf`). That redundancy makes plain OLS "
        "coefficients unstable. Ridge's L2 penalty stabilizes them at a small "
        "cost in bias. See `reports/figures/07_bootstrap_stability.png` and "
        "the notebook for the full evidence."
    )

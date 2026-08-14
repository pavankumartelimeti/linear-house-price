import pytest
from fastapi.testclient import TestClient

from src.config import resolve

MODEL_EXISTS = resolve("models", "production_model.joblib").exists()
pytestmark = pytest.mark.skipif(
    not MODEL_EXISTS, reason="Run `python -m src.models.train` before testing the API layer."
)

VALID_PAYLOAD = {
    "lot_area": 8500, "gr_liv_area": 1950, "total_bsmt_sf": 1000,
    "garage_area": 480, "year_built": 2005, "year_remod": 2015,
    "overall_qual": 7, "overall_cond": 6, "full_bath": 2, "half_bath": 1,
    "bedroom_abvgr": 3, "totrms_abvgrd": 7, "fireplaces": 1,
    "distance_to_downtown_km": 5.2, "school_rating": 8, "crime_index": 22.0,
    "median_income_area": 78.0, "has_pool": False,
    "neighborhood": "Lakeside", "house_style": "2Story",
}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_predict_returns_reasonable_price(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert 50_000 < body["predicted_price"] < 2_000_000
    assert body["predicted_price_formatted"].startswith("$")


def test_predict_works_without_optional_fields(client):
    payload = {k: v for k, v in VALID_PAYLOAD.items()
               if k not in ("total_bsmt_sf", "garage_area", "school_rating", "crime_index")}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["predicted_price"] > 0


def test_predict_rejects_invalid_neighborhood(client):
    payload = dict(VALID_PAYLOAD, neighborhood="Atlantis")
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_quality(client):
    payload = dict(VALID_PAYLOAD, overall_qual=99)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_batch_endpoint(client):
    resp = client.post("/predict-batch", json={"houses": [VALID_PAYLOAD, dict(VALID_PAYLOAD, neighborhood="Old Town")]})
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2


def test_model_info_endpoint(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert 0 < body["test_r2"] < 1
    assert body["best_model"]

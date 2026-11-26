import math

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def payload():
    return {
        "age_years": 0,
        "km_driven": 12000,
        "engine_cc": 800,
        "owners": 1,
        "service_records": None,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_accepts_boundary_values(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert math.isfinite(response.json()["resale_thousand_rs"])


@pytest.mark.parametrize("change", [
    {"age_years": -1},
    {"engine_cc": 801},
    {"unexpected_field": True},
])
def test_predict_rejects_invalid_input(client, payload, change):
    response = client.post("/predict", json={**payload, **change})
    assert response.status_code == 422

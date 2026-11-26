import math

import pytest
from fastapi.testclient import TestClient

from FastAPI import main as api
from app.main import app as alternate_app


VALID_REQUEST = {
    "age_years": 5,
    "km_driven": 20000,
    "engine_cc": 200,
    "owners": 1,
    "service_records": 3,
    "accidents": 0,
    "brand_tier": " BUD. ",
    "city_zone": "South",
}


class StubModel:
    def __init__(self):
        self.calls = 0
        self.features = None

    def predict(self, features):
        self.calls += 1
        self.features = features
        return [123.4]


@pytest.fixture
def api_client(monkeypatch):
    stub = StubModel()
    monkeypatch.setattr(api, "model", stub)
    return TestClient(api.app), stub


def test_health(api_client):
    client, _ = api_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_normalizes_inputs_and_returns_price(api_client):
    client, stub = api_client
    payload = {**VALID_REQUEST, "service_records": -999}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 123.4}
    assert list(stub.features.columns) == api.FEATURE_COLUMNS
    assert math.isnan(stub.features.loc[0, "service_records"])
    assert stub.features.loc[0, "brand_tier"] == "budget"


@pytest.mark.parametrize(
    "changes",
    [
        {"age_years": -1},
        {"engine_cc": 801},
        {"unexpected": "field"},
    ],
)
def test_invalid_requests_are_rejected_before_prediction(api_client, changes):
    client, stub = api_client

    response = client.post("/predict", json={**VALID_REQUEST, **changes})

    assert response.status_code == 422
    assert stub.calls == 0


@pytest.fixture(scope="module")
def alternate_client():
    with TestClient(alternate_app) as client:
        yield client


def test_alternate_app_health(alternate_client):
    response = alternate_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_alternate_app_predicts_boundary_values(alternate_client):
    payload = {
        "age_years": 0,
        "km_driven": 12000,
        "engine_cc": 800,
        "owners": 1,
        "service_records": None,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }

    response = alternate_client.post("/predict", json=payload)

    assert response.status_code == 200
    assert math.isfinite(response.json()["resale_thousand_rs"])


@pytest.mark.parametrize(
    "change",
    [
        {"age_years": -1},
        {"engine_cc": 801},
        {"unexpected_field": True},
    ],
)
def test_alternate_app_rejects_invalid_input(alternate_client, change):
    payload = {
        "age_years": 0,
        "km_driven": 12000,
        "engine_cc": 800,
        "owners": 1,
        "service_records": None,
        "accidents": 0,
        "brand_tier": "mid",
        "city_zone": "Central",
    }

    response = alternate_client.post("/predict", json={**payload, **change})

    assert response.status_code == 422

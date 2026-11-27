
import importlib
import io
import json
import logging
import math

import pytest
from fastapi.testclient import TestClient

from FastAPI import main as api
from app.main import app as alternate_app


GOOD = {
    "age_years": 5,
    "km_driven": 30000,
    "engine_cc": 200,
    "owners": 1,
    "service_records": 4,
    "accidents": 0,
    "brand_tier": "mid",
    "city_zone": "North",
}

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


@pytest.fixture(scope="module", params=["app.main", "FastAPI.main"])
def client(request):
    try:
        module = importlib.import_module(request.param)
    except ModuleNotFoundError:
        pytest.skip(f"{request.param} not found")

    with TestClient(module.app) as c:
        yield c


@pytest.fixture
def api_client(monkeypatch):
    stub = StubModel()
    monkeypatch.setattr(api, "model", stub)
    return TestClient(api.app), stub


# Basic API tests

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_ok(client):
    response = client.post("/predict", json=GOOD)
    assert response.status_code == 200


def test_engine_cc_above_800_rejected(client):
    response = client.post(
        "/predict",
        json={**GOOD, "engine_cc": 801}
    )
    assert response.status_code == 422


def test_negative_age_rejected(client):
    response = client.post(
        "/predict",
        json={**GOOD, "age_years": -1}
    )
    assert response.status_code == 422


# Prediction and input normalization tests

def test_predict_normalizes_inputs_and_returns_price(api_client):
    client, stub = api_client
    payload = {**VALID_REQUEST, "service_records": -999}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 123.4}
    assert list(stub.features.columns) == api.FEATURE_COLUMNS
    assert math.isnan(stub.features.loc[0, "service_records"])
    assert stub.features.loc[0, "brand_tier"] == "budget"


# JSON logging tests

def test_predict_logs_json_without_request_features(api_client):
    client, _ = api_client
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(api.JsonLogFormatter())
    api.logger.addHandler(handler)

    try:
        success = client.post("/predict", json=VALID_REQUEST)
        rejected = client.post(
            "/predict",
            json={**VALID_REQUEST, "engine_cc": 801}
        )
    finally:
        api.logger.removeHandler(handler)
        handler.close()

    events = [
        json.loads(line)
        for line in log_stream.getvalue().splitlines()
    ]

    assert success.status_code == 200
    assert rejected.status_code == 422
    assert len(events) == 2
    assert [event["status"] for event in events] == [
        "success", "rejected"
    ]
    assert all(event["service"] == "mp202-api" for event in events)
    assert all(event["route"] == "/predict" for event in events)
    assert all(event["request_id"] for event in events)
    assert len({event["request_id"] for event in events}) == 2

    assert success.headers["X-Request-ID"] == events[0]["request_id"]
    assert rejected.headers["X-Request-ID"] == events[1]["request_id"]

    assert all(event["duration_ms"] >= 0 for event in events)
    assert all(
        "engine_cc" not in event and "prediction" not in event
        for event in events
    )


# Invalid request validation

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

    response = client.post(
        "/predict",
        json={**VALID_REQUEST, **changes}
    )

    assert response.status_code == 422
    assert stub.calls == 0


# Alternate application tests

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

    response = alternate_client.post(
        "/predict",
        json={**payload, **change}
    )

    assert response.status_code == 422

import importlib

import pytest
from fastapi.testclient import TestClient

GOOD = {"age_years": 5, "km_driven": 30000, "engine_cc": 200, "owners": 1,
        "service_records": 4, "accidents": 0,
        "brand_tier": "mid", "city_zone": "North"}


@pytest.fixture(scope="module", params=["app.main", "FastAPI.main"])
def client(request):
    try:
        module = importlib.import_module(request.param)
    except ModuleNotFoundError:
        pytest.skip(f"{request.param} not found")
    with TestClient(module.app) as c:   # "with" runs startup so the model loads
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200


def test_engine_cc_above_800_rejected(client):
    assert client.post("/predict", json={**GOOD, "engine_cc": 801}).status_code == 422


def test_negative_age_rejected(client):
    assert client.post("/predict", json={**GOOD, "age_years": -1}).status_code == 422

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

FEATURE_COLUMNS = [
    "age_years",
    "km_driven",
    "engine_cc",
    "owners",
    "service_records",
    "accidents",
    "brand_tier",
    "city_zone",
]
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(ge=0)
    km_driven: float
    engine_cc: float = Field(le=800)
    owners: int
    service_records: float | None
    accidents: float | None
    brand_tier: str
    city_zone: str


class PredictionResponse(BaseModel):
    prediction: float


app = FastAPI(title="MP202 Two-Wheeler Price API")


class JsonLogFormatter(logging.Formatter):
    def format(self, record):
        event = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": "mp202-api",
            "event": getattr(record, "event_name", "prediction_request"),
            "status": getattr(record, "status", "unknown"),
            "route": getattr(record, "route", "/predict"),
            "status_code": getattr(record, "status_code", None),
            "request_id": getattr(record, "request_id", None),
            "duration_ms": getattr(record, "duration_ms", None),
        }
        error_type = getattr(record, "error_type", None)
        if error_type is not None:
            event["error_type"] = error_type
        return json.dumps(event, separators=(",", ":"), allow_nan=False)


logger = logging.getLogger("mp202.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not any(
    getattr(handler, "_mp202_json_handler", False)
    for handler in logger.handlers
):
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(JsonLogFormatter())
    stdout_handler._mp202_json_handler = True
    logger.addHandler(stdout_handler)


@app.middleware("http")
async def log_prediction_requests(request: Request, call_next):
    if request.url.path != "/predict" or request.method != "POST":
        return await call_next(request)

    request_id = uuid.uuid4().hex[:16]
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "prediction request failed",
            extra={
                "event_name": "prediction_request",
                "status": "error",
                "route": "/predict",
                "status_code": 500,
                "request_id": request_id,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "error_type": type(exc).__name__,
            },
        )
        raise

    response.headers["X-Request-ID"] = request_id
    logger.log(
        logging.INFO if response.status_code < 400 else logging.WARNING,
        "prediction request completed",
        extra={
            "event_name": "prediction_request",
            "status": "success" if response.status_code < 400 else "rejected",
            "route": "/predict",
            "status_code": response.status_code,
            "request_id": request_id,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    values = request.model_dump()
    if values["service_records"] is None or values["service_records"] == -999:
        values["service_records"] = np.nan
    if values["accidents"] is None:
        values["accidents"] = np.nan

    brand = values["brand_tier"].strip().lower().replace(".", "")
    values["brand_tier"] = {"bud": "budget", "pre": "premium"}.get(brand, brand)
    features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = model.predict(features)[0]
    return PredictionResponse(prediction=float(prediction))

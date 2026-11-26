from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
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

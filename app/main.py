from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_years: float = Field(..., ge=0)
    km_driven: float
    engine_cc: float = Field(..., le=800)
    owners: int
    service_records: int | None
    accidents: int | None
    brand_tier: str
    city_zone: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(body: PredictionRequest):
    features = pd.DataFrame([body.model_dump()]).fillna(float("nan"))
    prediction = app.state.model.predict(features)[0]
    return {"resale_thousand_rs": float(prediction)}

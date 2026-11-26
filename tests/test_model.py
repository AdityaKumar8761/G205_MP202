import numpy as np
import pandas as pd

from mp202.train import build_pipeline, load_params


def test_params_yaml_is_read():
    cfg = load_params()
    assert cfg["mlflow"]["registered_model_name"] == "mp202-price-model"


def test_pipeline_fits_and_predicts_with_missing_values():
    cfg = load_params()
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame({
        "age_years": rng.uniform(1, 12, n),
        "km_driven": rng.integers(1000, 90000, n),
        "engine_cc": rng.integers(100, 600, n),
        "owners": rng.integers(1, 4, n),
        "service_records": rng.integers(0, 15, n).astype(float),
        "accidents": rng.integers(0, 3, n).astype(float),
        "brand_tier": rng.choice(["budget", "mid", "premium"], n),
        "city_zone": rng.choice(["North", "South", "Central"], n),
    })
    X.loc[0, "service_records"] = np.nan          # NaN must not crash
    y = 100 - X["age_years"] * 5 + rng.normal(0, 1, n)
    run = {"model": "random_forest", "n_estimators": 10, "max_depth": 3}
    pipe = build_pipeline(cfg, run).fit(X, y)
    preds = pipe.predict(X.head(3))
    assert preds.shape == (3,)
    assert np.isfinite(preds).all()

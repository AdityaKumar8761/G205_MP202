"""Task 5: MLflow tracking + registry. From repo root: PYTHONPATH=src python -m mp202.train"""
import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mp202.cleaning import clean_data


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_pipeline(cfg, run):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), cfg["features"]["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["features"]["categorical"]),
    ])
    rs = cfg["training"]["random_state"]
    if run["model"] == "random_forest":
        reg = RandomForestRegressor(n_estimators=run["n_estimators"],
                                    max_depth=run["max_depth"], random_state=rs)
    else:
        reg = GradientBoostingRegressor(n_estimators=run["n_estimators"],
                                        learning_rate=run["learning_rate"], random_state=rs)
    return Pipeline([("pre", pre), ("model", reg)])


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    cfg = load_params()
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    target = cfg["features"]["target"]

    train = clean_data(pd.read_csv(cfg["data"]["train_path"]))
    later = clean_data(pd.read_csv(cfg["data"]["later_path"]))
    data_hash = sha256_of(cfg["data"]["train_path"])  # ORIGINAL train.csv

    X_tr, X_va, y_tr, y_va = train_test_split(
        train[feats], train[target],
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results = []
    for run_cfg in cfg["training"]["runs"]:
        with mlflow.start_run(run_name=run_cfg["name"]) as run:
            pipe = build_pipeline(cfg, run_cfg).fit(X_tr, y_tr)
            mae_valid = mean_absolute_error(y_va, pipe.predict(X_va))
            mae_later = mean_absolute_error(later[target], pipe.predict(later[feats]))
            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_params({k: v for k, v in run_cfg.items() if k != "name"})
            mlflow.log_metric("mae_valid", mae_valid)
            mlflow.log_metric("mae_later", mae_later)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
            results.append((mae_valid, run.info.run_id, run_cfg["name"]))
            print(f"{run_cfg['name']}: mae_valid={mae_valid:.3f} mae_later={mae_later:.3f}")

    best_mae, best_run_id, best_name = min(results)  # choose by mae_valid, not by later
    name = cfg["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best_run_id}/model", name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)

    # refit the winner on all cleaned train data -> file the API will load
    best_cfg = next(r for r in cfg["training"]["runs"] if r["name"] == best_name)
    final = build_pipeline(cfg, best_cfg).fit(train[feats], train[target])
    Path(cfg["training"]["model_path"]).parent.mkdir(exist_ok=True)
    joblib.dump(final, cfg["training"]["model_path"])

    print("\n--- paste into reports/findings.yaml -> tracking ---")
    print("experiment_name:", cfg["mlflow"]["experiment_name"])
    print("best_run_id:", best_run_id)
    print("registered_model_name:", name)
    print("champion_version:", mv.version)


if __name__ == "__main__":
    main()

# G205 — MP202: Used Two-Wheeler Price (Kochi)

**SCSE3040 MLOps Mini Project** · Group **G205** · Problem **MP202**
Faculty: Saurabh Shrivastava · Batch: 49

An end-to-end MLOps pipeline that predicts the resale price (`resale_thousand_rs`, in thousand rupees) of a used two-wheeler for the *Kochi Second-Hand Bike Market*. It is a regression problem, measured with **MAE**.

## Group members

| Enrolment no. | Name | Main parts (see [Task ownership](#task-ownership)) |
|---|---|---|
| S24CSEU1905 | Aditya Sinha Kumar | Git, DVC, FastAPI service, Dockerfile, GitHub Actions |
| S24CSEU1920 | Aradhya Sharma | Data problem hunt, data cleaning |
| S24CSEU1910 | Muskan Sundrani | MLflow tracking & registry, tests |
| S24CSEU1899 | Ansh Sharma | Kubernetes, JSON logging |

## What the pipeline does

1. **Version data** with DVC (`data.dvc`, remote in `.dvc/config`).
2. **Find and fix the 7 kinds of data problems** in `train.csv` (`src/mp202/cleaning.py`).
3. **Train and track** 3 models with MLflow, register the best one and give it the `champion` alias (`src/mp202/train.py`).
4. **Serve** the champion with FastAPI (`POST /predict`, `GET /health`) with the client's validation rules.
5. **Containerise** with Docker and **deploy** on Kubernetes (2 replicas, probes on `/health`).
6. **Test automatically** with pytest on every push (GitHub Actions).
7. **Log** every prediction request as one JSON line on stdout.
8. **Monitor drift** (PSI), compare v1 vs v2 on new data, apply the promotion rule, and **predict the September holdout**.

## Repository layout

```
README.md                     this file
GROUP.md / PROBLEM.md         group details and the problem statement
STUDENT_HANDBOOK.md           course handbook
tasks                         task list: owner + status + launch commands
params.yaml                   paths, features, MLflow and training settings
requirements.txt              pinned dependencies
src/mp202/
    cleaning.py               clean_data(): fixes all data problems
    train.py                  MLflow tracking, registry, final model.joblib
app/main.py                   FastAPI service (basic version)
FastAPI/main.py               FastAPI service with JSON logging + input normalisation (used by Docker)
tests/                        test_cleaning.py, test_model.py, test_api.py (uses TestClient)
models/model.joblib           champion model loaded by the API
mp202-champion-v2/            exported MLflow model (champion, version 2)
mlflow.db                     MLflow tracking database
data.dvc, .dvc/               DVC tracking (data itself is NOT in git)
notebooks/data_analysis.ipynb exploration of train.csv
reports/findings.yaml         filled findings form (data issues, tracking, monitoring)
predictions/holdout_predictions.csv   final predictions (record_id, prediction)
Task_9_Kubernetes/            Kubernetes notebook + work/ (deployment.yaml, service.yaml)
Task_10_Logging/              JSON logging notebook + sample logs.txt
Dockerfile, .dockerignore     container build
.github/workflows/ci.yml      CI: runs pytest on every push
```

## Task ownership

Tasks are done **in this order** (a different order can break the project). Everything is marked done.

| # | Task | Owner | Status |
|---|---|---|---|
| 1 | Git init | Aditya | ✅ Done |
| 2 | DVC | Aditya | ✅ Done |
| 3 | Find the data problems in `train.csv` | Aradhya | ✅ Done |
| 4 | Cleaning data | Aradhya | ✅ Done |
| 5 | MLflow tracking and registry | Muskan | ✅ Done |
| 6 | Tests | Muskan | ✅ Done |
| 7 | FastAPI service | Aditya | ✅ Done |
| 8 | Dockerfile | Aditya | ✅ Done |
| 8.5 | Green GitHub Action (workflow created) | Aditya | ✅ Done |
| 9 | Kubernetes | Ansh | ✅ Done |
| 10 | Logging | Ansh | ✅ Done |
| 11 | Drift, retraining, holdout predictions, `findings.yaml` | Group | ✅ Done |

## Data problems found

Detected from our own code on `train.csv`; details in `reports/findings.yaml`.

| Problem | Present? | Details | Fix in `clean_data()` |
|---|---|---|---|
| Leaking column | Yes | `final_price_from_rc_transfer` (only known after the sale; missing from holdout) | Column dropped |
| Sentinel value | Yes | `service_records` = `-999`, 38 rows | Replaced with `NaN` |
| Stuck sensor | Yes | `accidents`, 2026-04-11 to 2026-04-29 | Values in that window set to `NaN` |
| Spelling variants | Yes | `brand_tier` (3 true levels: budget / mid / premium) | Strip, lowercase, remove `.`, map `bud`→`budget`, `pre`→`premium` |
| Unit mix-up | No | — | — |
| Duplicate rows | No | — | — |
| Target typos | No | — | — |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
dvc pull                           # fetches data/ from the DVC remote (needs access to the S3 bucket)
```

## How to run

**Train, track and register models** (from the repo root):

```bash
PYTHONPATH=src python -m mp202.train
```

This runs the 3 configurations in `params.yaml` (2 random forests + 1 gradient boosting), logs `data_sha256`, `mae_valid` and `mae_later` for each, registers the best one by `mae_valid`, sets the `champion` alias, and saves the refit model to `models/model.joblib`.

**Run tests:**

```bash
python -m pytest -q
```

Tests build their own small data, so they do not need the CSV files.

**Run the API locally:**

```bash
uvicorn FastAPI.main:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age_years": 5, "km_driven": 30000, "engine_cc": 200, "owners": 1,
  "service_records": 4, "accidents": 0, "brand_tier": "mid", "city_zone": "North"
}'
# {"prediction": ...}
```

Client rules enforced in the pydantic schema (invalid input returns **422**):

- `engine_cc` above 800 is rejected.
- `age_years` below 0 is rejected.

`GET /health` returns `{"status": "ok"}`.

## Docker

```bash
docker build -t mp202-api:1.0 .
docker run -p 8000:8000 mp202-api:1.0
```

## Kubernetes (minikube)

Deployment `mp202-api` (2 replicas, image `mp202-api:1.0`, port 8000) and Service `mp202-svc` (port 80 → 8000). Readiness and liveness probes both hit `/health`; liveness waits longer (30 s vs 5 s).

**Launch:**

```bash
minikube start --driver=docker
docker build -t mp202-api:1.0 .
minikube image load mp202-api:1.0

kubectl apply -f Task_9_Kubernetes/work/deployment.yaml
kubectl apply -f Task_9_Kubernetes/work/service.yaml
kubectl rollout status deployment/mp202-api

kubectl port-forward service/mp202-svc 8080:80
```

The API is then available at `http://localhost:8080`.

**Stop:**

```bash
kubectl delete -f Task_9_Kubernetes/work/service.yaml -f Task_9_Kubernetes/work/deployment.yaml
minikube stop
```

## Logging

Each `POST /predict` writes one JSON line to stdout (never to a file, and never with the request's feature values):

```json
{"timestamp":"2026-11-27T07:43:46.767+00:00","level":"INFO","service":"mp202-api","event":"prediction_request","status":"success","route":"/predict","status_code":200,"request_id":"7a4858ccc19f49a6","duration_ms":54.8}
```

Rejected requests (422) are logged with `status: "rejected"` at WARNING level. A sample is in `Task_10_Logging/logs.txt`.

## CI

`.github/workflows/ci.yml` runs on every `push`: it installs `requirements.txt` and runs `python -m pytest -q`.

## Monitoring and retraining results

From `reports/findings.yaml`:

| Check | Result |
|---|---|
| Most drifted column (PSI, train vs later) | `engine_cc` (PSI ≈ 5.22); every other column below 0.04 |
| MAE of v1 on `later.csv` per `city_zone` | Central ≈ 12.86 · South ≈ 11.16 · **North ≈ 49.25** |
| Worst-served group | **North** |
| v1 MAE on the last 30% of `later.csv` | ≈ 34.64 |
| v2 MAE on the last 30% (trained on train + first 70% of later) | ≈ 14.77 |
| Improvement | ≈ 57% lower (the rule needs at least 12%) |
| Decision | **Promote v2** → v2 is the `@champion` (version 2) in the MLflow registry |

Final predictions for the 250 September holdout rows are in `predictions/holdout_predictions.csv` (`record_id,prediction`).

## Tech stack

Python · pandas · scikit-learn · MLflow (SQLite backend) · DVC · FastAPI + pydantic · uvicorn · pytest · Docker · Kubernetes (minikube) · GitHub Actions

## Notes

- Never open or re-save `data/train.csv` in Excel; that changes its hash and the DVC/MLflow data fingerprint stops matching.
- Task order matters: follow the order in the table above if you rebuild the project from scratch.
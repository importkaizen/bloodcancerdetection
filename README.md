# HemaSight – CBC Screening Research and Blood-Pattern Analysis

[Python 3.10+](https://www.python.org/downloads/)
[License: MIT](https://opensource.org/licenses/MIT)

HemaSight combines a reproducible CBC screening benchmark with an experimental application for longitudinal blood-pattern analysis. The research workflow compares five common CBC measurements with an expanded fifteen-feature set across the released LeukoAlert hospital cohorts. The project has not established early cancer prediction, clinical efficacy, or diagnostic suitability.

## Research workflow

Start with [the research guide](docs/RESEARCH.md) and [the first experiment protocol](docs/EXPERIMENT_PROTOCOL.md). The original application setup follows below.

The [first completed benchmark](docs/FIRST_RESULTS.md) includes eight experiments on 446,663 released records, full hospital-level comparisons, and reproducibility metadata.

See [application changes and verification](docs/APPLICATION_TESTING.md) for durable queue delivery, retry-safe ingestion, corrected chart timelines, and the latest local checks.

```bash
# Python 3.11; create and activate a virtual environment first
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
python -m hemasight.research.leukoalert download --output data/raw/LeukoAlert-project.zip
python -m hemasight.research.leukoalert prepare --archive data/raw/LeukoAlert-project.zip --output data/processed/leukoalert-v1
python -m hemasight.research.benchmark data/processed/leukoalert-v1/features.csv --output runs/development-v1 --models logistic rf xgboost
```

The benchmark preserves hospital cohorts, learns preprocessing from development data only, and saves immutable models and aggregate evaluation reports. External/test evaluation requires `--evaluate-holdouts` after fixing the experiment. Patient overlap cannot be verified from the released sample IDs; the data do not include the dates needed to establish advance prediction. Raw data and model artifacts are excluded from Git.

Longitudinal workers now use feature version `v2`: the latest five visits and trends per elapsed day. Existing `v1` models require retraining against matching features. PyTorch is optional and installed through `.[deep-learning]` for the separate experimental neural models.

---

## Features

- **Distributed pipeline**: API → RabbitMQ → Celery workers → ML inference
- **Time-series features**: Rolling averages, trend slopes, variance, abnormal ratios
- **ML models**: Random Forest, XGBoost, LSTM (optional)
- **Anomaly detection**: Isolation Forest for outlier detection
- **TimescaleDB**: Optimized storage for time-series blood test data
- **React dashboard**: Patient list, blood history charts, risk score trends

---

## Architecture

```
Client (Doctor/Researcher)
    ↓
API Gateway (FastAPI)
    ↓
Data Ingestion (POST /blood-test)
    ↓
Message Queue (RabbitMQ)
    ↓
Feature Workers (Celery) → ML Risk Engine + Anomaly Detection
    ↓
PostgreSQL + TimescaleDB
    ↓
React Dashboard
```

---

## Tech Stack


| Layer    | Technology                     |
| -------- | ------------------------------ |
| Backend  | Python, FastAPI                |
| ML       | PyTorch, scikit-learn, XGBoost |
| Pipeline | RabbitMQ (optional Kafka)      |
| Database | PostgreSQL, TimescaleDB        |
| Workers  | Celery                         |
| Frontend | React, Vite, Recharts          |
| DevOps   | Docker, Docker Compose         |


---

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- Optional for local dev: Python 3.11 (tested), Node 22

### Run the full stack

From the **repository root**:

```bash
cd hemasight/docker
docker compose up -d
```

> Use `docker-compose` if you have the standalone Compose V1.


| Service     | URL                                                            |
| ----------- | -------------------------------------------------------------- |
| API         | [http://localhost:8000](http://localhost:8000)                 |
| API docs    | [http://localhost:8000/docs](http://localhost:8000/docs)       |
| Dashboard   | [http://localhost:3000](http://localhost:3000)                 |
| RabbitMQ UI | [http://localhost:15672](http://localhost:15672) (hemasight/hemasight in local Compose) |


### Ingest a blood test

```bash
curl -X POST http://localhost:8000/blood-test \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","date":"2025-05-01","wbc":7.2,"rbc":4.8,"platelets":210,"hemoglobin":13.5,"lymphocytes":40}'
```

Response: `202 Accepted` with `blood_test_id` and `patient_id`. The test and delivery event are saved together; the dispatcher retries publication during queue outages. Celery computes features, then schedules risk and anomaly scoring. Use an `Idempotency-Key` header when retrying imports to reuse the original test instead of adding a duplicate.

---

## Train the risk model (optional)

Use the sample CSV or your own data with columns matching the feature schema and a `label` column (0 = normal, 1 = at-risk):

```bash
pip install -e .
python -m hemasight.ml.model_training hemasight/ml/data/sample_training_data.csv --model-type xgboost
```

Copy the generated `risk_model.pkl`, `scaler.pkl`, and `config.json` into `hemasight/ml/models/` or mount that directory in the API/workers containers.

### Train the anomaly model (optional)

```python
from hemasight.ml.anomaly import fit_isolation_forest
import numpy as np

X = np.random.randn(100, 13)  # 100 samples, 13 features
fit_isolation_forest(X)
```

---

## API Reference


| Method | Endpoint                     | Description                      |
| ------ | ---------------------------- | -------------------------------- |
| POST   | `/blood-test`                | Ingest blood test (202 Accepted) |
| GET    | `/patients`                  | List all patients                |
| GET    | `/patients/{id}/blood-tests` | Blood tests for a patient        |
| GET    | `/patients/{id}/risk-scores` | Risk scores for a patient        |
| GET    | `/health`                    | Health check                     |


Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Project Structure

```
hemasight/
├── api/              # FastAPI app, routes, schemas
├── ml/               # model_training, inference, anomaly, LSTM
├── workers/          # Celery feature worker + risk/anomaly tasks
├── data_pipeline/    # RabbitMQ producer & consumer
├── db/               # SQLAlchemy models, migrations
├── frontend/         # React dashboard (Vite)
└── docker/           # Docker Compose, Dockerfiles
```

---

## Datasets

- **Sample**: `hemasight/ml/data/sample_training_data.csv`
- **Research**: see [LeukoAlert preparation and dataset limitations](docs/RESEARCH.md). Longitudinal work requires an appropriate patient-linked dataset and a defined outcome timeline; mapping feature names alone is insufficient.

---

## Optional: Kafka

The default broker is RabbitMQ. Kafka producer/consumer abstractions exist in `hemasight/data_pipeline/`; extend `producer.py` and `consumer.py` for Kafka-based deployments.

---

## License

MIT. Use only for research and non-diagnostic purposes. Comply with local regulations and dataset licenses.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---


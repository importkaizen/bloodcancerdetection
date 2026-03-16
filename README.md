# HemaSight – Distributed AI for Early Blood Cancer Risk Detection

HemaSight is a **distributed AI platform** for detecting early abnormal patterns in CBC (complete blood count) time-series data, focused on hematologic abnormalities (e.g. Leukemia, Lymphoma, Multiple Myeloma). The system does **not** provide a diagnosis; it surfaces **pattern drift** that may indicate early hematologic abnormalities, e.g.:

> *"Your blood patterns show unusual drift that may indicate early hematologic abnormalities."*

## Architecture

- **Client** (Doctor / Researcher) → **API Gateway** (FastAPI) → **Data Ingestion** (POST /blood-test)
- **Message Queue** (RabbitMQ) → **Feature Workers** (Celery) → **ML Risk Engine** + **Anomaly Detection**
- **Database**: PostgreSQL + TimescaleDB (time-series)
- **Dashboard**: React app for patient list, blood history charts, and risk score over time

## Tech Stack

| Layer     | Choice                |
|----------|------------------------|
| Backend  | Python, FastAPI        |
| ML       | PyTorch, scikit-learn, XGBoost |
| Pipeline | RabbitMQ (optional Kafka) |
| DB       | PostgreSQL, TimescaleDB |
| Workers  | Celery                 |
| Frontend | React, Recharts        |
| DevOps   | Docker, docker-compose |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) Python 3.10+ and Node 18+ for local dev

### Run full stack with Docker

From the **repository root** (parent of `hemasight/`):

```bash
cd hemasight/docker
docker-compose up -d
```

- **API**: http://localhost:8000  
- **API docs**: http://localhost:8000/docs  
- **Dashboard**: http://localhost:3000  
- **RabbitMQ management**: http://localhost:15672 (guest/guest)

### Ingest a blood test

```bash
curl -X POST http://localhost:8000/blood-test \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","date":"2025-05-01","wbc":7.2,"rbc":4.8,"platelets":210,"hemoglobin":13.5,"lymphocytes":40}'
```

Response: `202 Accepted` with `blood_test_id` and `patient_id`. The payload is stored, published to RabbitMQ, and processed by Celery workers (features → risk score → anomaly score).

### Train the risk model (optional)

Use sample or your own CSV with columns matching the feature schema and a `label` column (0 = normal, 1 = at-risk):

```bash
pip install -e .
python -m hemasight.ml.model_training hemasight/ml/data/sample_training_data.csv --model-type xgboost
```

Copy the generated `risk_model.pkl`, `scaler.pkl`, and `config.json` into the API/workers container or mount `hemasight/ml/models` so the risk pipeline can run.

### Train anomaly model (optional)

From Python or a script: build a feature matrix `X` (e.g. from your DB or CSV), then:

```python
from hemasight.ml.anomaly import fit_isolation_forest
import numpy as np
X = np.random.randn(100, 13)  # 100 samples, 13 features
fit_isolation_forest(X)
```

## Datasets

- Use **hemasight/ml/data/sample_training_data.csv** for a minimal example.
- For real research: ALL (leukemia), MIMIC (with access), or Kaggle CBC datasets. Map columns to the schema (`wbc`, `rbc`, `platelets`, `hemoglobin`, `lymphocytes`, plus derived features and a `label` column). Do not commit PHI.

## Project layout

```
hemasight/
  api/           FastAPI app, routes, schemas
  ml/            model_training, inference, anomaly, LSTM option
  workers/       Celery feature worker + risk/anomaly tasks
  data_pipeline/ RabbitMQ producer & consumer
  db/            SQLAlchemy models, migrations
  frontend/      React dashboard (Vite)
  docker/        docker-compose, Dockerfiles
```

## API

- `POST /blood-test` – Ingest blood test (202 Accepted)
- `GET /patients` – List patients
- `GET /patients/{id}/blood-tests` – Blood tests for a patient
- `GET /patients/{id}/risk-scores` – Risk scores for a patient
- `GET /health` – Health check

## Optional: Kafka

The default message broker is RabbitMQ. An optional Kafka producer/consumer abstraction can be added (e.g. via `BROKER=kafka` and `confluent-kafka`) for a “large streams” deployment; see `hemasight/data_pipeline/` and extend `producer.py` / `consumer.py` accordingly.

## License

MIT (or your choice). Use only for research and non-diagnostic purposes; comply with local regulations and dataset licenses.

## Resume bullet

> Built a distributed AI platform for early hematologic cancer risk detection using time-series CBC analysis with FastAPI, RabbitMQ, PyTorch/scikit-learn, TimescaleDB, Celery workers, and Docker.

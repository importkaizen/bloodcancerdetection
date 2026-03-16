# HemaSight – Distributed AI for Early Blood Cancer Risk Detection

[Python 3.10+](https://www.python.org/downloads/)
[License: MIT](https://opensource.org/licenses/MIT)

HemaSight is a **distributed AI platform** for detecting early abnormal patterns in CBC (complete blood count) time-series data. It targets hematologic abnormalities such as Leukemia, Lymphoma, and Multiple Myeloma. The system does **not** provide a diagnosis; it surfaces **pattern drift** that may indicate early hematologic abnormalities.

> *"Your blood patterns show unusual drift that may indicate early hematologic abnormalities."*

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
- Optional for local dev: Python 3.10+, Node 18+

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
| RabbitMQ UI | [http://localhost:15672](http://localhost:15672) (guest/guest) |


### Ingest a blood test

```bash
curl -X POST http://localhost:8000/blood-test \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","date":"2025-05-01","wbc":7.2,"rbc":4.8,"platelets":210,"hemoglobin":13.5,"lymphocytes":40}'
```

Response: `202 Accepted` with `blood_test_id` and `patient_id`. The payload is stored, published to RabbitMQ, and processed by Celery workers (features → risk score → anomaly score).

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
- **Research**: ALL (leukemia), MIMIC (with access), or Kaggle CBC datasets. Map columns to the schema (`wbc`, `rbc`, `platelets`, `hemoglobin`, `lymphocytes`, plus derived features and a `label` column). Do not commit PHI.

---

## Optional: Kafka

The default broker is RabbitMQ. Kafka producer/consumer abstractions exist in `hemasight/data_pipeline/`; extend `producer.py` and `consumer.py` for Kafka-based deployments.

---

## License

---

## License

MIT. Use only for research and non-diagnostic purposes. Comply with local regulations and dataset licenses.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not provide medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

---


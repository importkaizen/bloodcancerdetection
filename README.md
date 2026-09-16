# HemaSight — CBC Screening Research and Blood-Pattern Analysis

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/pytest-118%20passed-brightgreen)
![Status](https://img.shields.io/badge/Status-research%20only-lightgrey)

HemaSight combines a reproducible complete blood count (CBC) screening benchmark with an
experimental application for longitudinal blood-pattern analysis. The research workflow
compares five common CBC measurements against an expanded fifteen-feature set across the
released LeukoAlert hospital cohorts. The application track provides ingestion, queuing,
feature computation, and a review dashboard.

> **Scope notice.** This project has not established early cancer prediction, clinical
> efficacy, or diagnostic suitability. It is research software and is not a medical device.
> See [Interpretation limits](#interpretation-limits) and [Disclaimer](#disclaimer).

---

## Contents

- [Research track](#research-track)
- [Application track](#application-track)
- [Results summary](#results-summary)
- [Repository layout](#repository-layout)
- [Installation and verification](#installation-and-verification)
- [Data provenance](#data-provenance)
- [API reference](#api-reference)
- [Interpretation limits](#interpretation-limits)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Research track

Start with the [research guide](docs/RESEARCH.md), the
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md), and the
[first results](docs/FIRST_RESULTS.md).

```bash
# Python 3.11; create and activate a virtual environment first
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
python -m hemasight.research.leukoalert download --output data/raw/LeukoAlert-project.zip
python -m hemasight.research.leukoalert prepare --archive data/raw/LeukoAlert-project.zip --output data/processed/leukoalert-v1
python -m hemasight.research.benchmark data/processed/leukoalert-v1/features.csv --output runs/development-v1 --models logistic rf xgboost
```

Design properties of the benchmark:

- Hospital cohorts are preserved; preprocessing (median imputation, logistic-regression
  scaling) is learned from development data only.
- Fixed hyperparameters, seed 42. No feature selection, oversampling, recalibration,
  or hyperparameter search.
- The operating threshold targets 95% specificity on pooled validation negatives and is
  frozen before any external/test evaluation (`--evaluate-holdouts`).
- Runs are immutable: each output directory contains `manifest.json`, `metrics.json`,
  `leaderboard.md`, and separate model bundles with fitted preprocessing, feature order,
  and threshold. Previous runs are never overwritten.
- Reports contain aggregate results only. Raw records, artifact paths, and fitted models
  are not served by the API.

Supplementary analyses: dataset distribution profiles
(`python -m hemasight.research.profile ...`) and a five-fold development
cross-validation diagnostic (`python -m hemasight.research.cross_validation ...`),
both documented in the [research guide](docs/RESEARCH.md). Raw data and model
artifacts are excluded from version control.

---

## Application track

The application is an experimental ingestion-to-review pipeline: FastAPI ingestion,
RabbitMQ dispatch with durable delivery, Celery feature workers (feature version `v2`:
latest five visits, trends per elapsed day), ML risk and anomaly scoring, TimescaleDB
storage, and a React dashboard with patient timelines, a research-benchmarks view, and
a dataset-comparison view. Details and verification scope are in
[application testing notes](docs/APPLICATION_TESTING.md).

Existing `v1` models are not numerically compatible with `v2` features and require
retraining. PyTorch is optional (`.[deep-learning]`) for the separate experimental
neural models. Benchmark models stay offline; the research workflow never automatically
publishes a model to the API.

### Run the full stack

Prerequisites: Docker and Docker Compose. Optional for local development:
Python 3.11 (tested), Node 22.

From the repository root:

```bash
cd hemasight/docker
docker compose up -d
```

(Use `docker-compose` if you have the standalone Compose V1.)

| Service     | URL |
| ----------- | --- |
| API         | <http://localhost:8000> |
| API docs    | <http://localhost:8000/docs> |
| Dashboard   | <http://localhost:3000> |
| RabbitMQ UI | <http://localhost:15672> (`hemasight` / `hemasight` in local Compose) |

### Ingest a blood test

```bash
curl -X POST http://localhost:8000/blood-test \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"P001","date":"2025-05-01","wbc":7.2,"rbc":4.8,"platelets":210,"hemoglobin":13.5,"lymphocytes":40}'
```

Response: `202 Accepted` with `blood_test_id` and `patient_id`. The test and its
delivery event are saved atomically; the dispatcher retries publication during queue
outages. Workers compute features, then schedule risk and anomaly scoring. Pass an
`Idempotency-Key` header when retrying imports to reuse the original test instead of
creating a duplicate.

### Train the risk model (optional)

Use the sample CSV or your own data with columns matching the feature schema and a
`label` column (0 = normal, 1 = at-risk):

```bash
pip install -e .
python -m hemasight.ml.model_training hemasight/ml/data/sample_training_data.csv --model-type xgboost
```

Copy the generated `risk_model.pkl`, `scaler.pkl`, and `config.json` into
`hemasight/ml/models/` or mount that directory in the API/worker containers.

---

## Results summary

First fixed experiment (completed 2026-09-13, LeukoAlert v1): eight prespecified
combinations on 446,663 released records — 26,922 development records for fitting,
211,612 validation records for threshold selection, 149,648 external and 58,481 test
records for frozen evaluation. Full evidence: [first results](docs/FIRST_RESULTS.md),
[cohort leaderboard](docs/results/holdout-v1/leaderboard.md),
[metrics](docs/results/holdout-v1/metrics.json), [manifest](docs/results/holdout-v1/manifest.json).

| Fifteen-feature model | External AUROC | External sensitivity | External specificity | Test AUROC | Test sensitivity | Test specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.9205 | 62.88% | 95.90% | 0.9308 | 67.38% | 95.36% |
| Random Forest | 0.9643 | 79.88% | 95.64% | 0.9699 | 87.29% | 94.53% |
| XGBoost | 0.9648 | 79.08% | 95.91% | 0.9689 | 87.16% | 94.76% |

These are descriptive record-level comparisons; statistical significance has not been
established. Hospital variation is substantial (e.g. expanded-XGBoost specificity
96.93% at external site E versus 85.91% at site G), and calibration remains a research
problem. The external and test outcomes have been inspected and must not be treated as
untouched cohorts for future model selection.

---

## Repository layout

```
hemasight/
├── api/              # FastAPI app, routes (incl. research), schemas
├── research/         # leukoalert, benchmark, cross_validation, profile
├── ml/               # model_training, inference, anomaly, features, LSTM
├── workers/          # Celery feature worker, risk/anomaly tasks
├── data_pipeline/    # RabbitMQ producer/consumer, outbox
├── db/               # SQLAlchemy models, migrations
├── frontend/         # React dashboard (Vite)
└── docker/           # Docker Compose, Dockerfiles, nginx
docs/
├── RESEARCH.md             # Research guide
├── EXPERIMENT_PROTOCOL.md  # Prespecified protocol
├── FIRST_RESULTS.md        # First benchmark results
├── APPLICATION_TESTING.md  # Application verification notes
└── results/holdout-v1/     # Saved aggregate reports (manifest, metrics, …)
scripts/              # Container-stack and queue-recovery checks
tests/                # Automated test suite
```

---

## Installation and verification

Python 3.11 is the tested environment. The pinned `requirements.lock` was resolved on
Windows; CI runs the Python suite on Windows/Python 3.11 and the dashboard
lint/build on Ubuntu/Node 22 (see `.github/workflows/research-tests.yml`).

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

The suite (118 passed, 1 skipped on the reference run) covers dataset preparation,
cohort boundaries, development-only preprocessing, threshold selection, aggregate
reporting, longitudinal feature windows, artifact version checks, queue delivery, and
the research API. The Docker application stack and optional PyTorch routines are not
part of the default test run.

---

## Data provenance

Source: Liu, Shilong (2026), [LeukoAlert-project, version 1](https://data.mendeley.com/datasets/vc7kwnyppz/1),
DOI `10.17632/vc7kwnyppz.1`, CC BY 4.0. Accompanying paper:
<https://www.nature.com/articles/s41746-026-03016-3>. Cite the source when publishing
analyses. Download and preparation verify the archive SHA-256; preparation reads only
explicitly named CSV entries, never extracts or executes the supplied research code,
and never overwrites existing dataset directories.

- **Sample fixture:** `hemasight/ml/data/sample_training_data.csv` (smoke tests only).
- **Research data:** prepared via `hemasight.research.leukoalert`; see the
  [research guide](docs/RESEARCH.md) for canonical cohorts, the five- and
  fifteen-feature definitions, and unit conversions.

---

## API reference

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| POST | `/blood-test` | Ingest blood test (202 Accepted) |
| GET | `/patients` | List all patients |
| GET | `/patients/{id}/blood-tests` | Blood tests for a patient |
| GET | `/patients/{id}/risk-scores` | Risk scores for a patient |
| GET | `/research/runs` | Catalog of validated aggregate reports |
| GET | `/research/runs/{run_id}` | Aggregate report (`?download=true` to export) |
| GET | `/research/runs/{run_id}/profile` | Dataset distribution profile |
| GET | `/research/runs/{run_id}/cross-validation` | Development cross-validation aggregates |
| GET | `/health` | Health check |

Interactive docs: <http://localhost:8000/docs>. Research reads served aggregates only;
invalid reports are omitted from the catalog, and raw records, artifact paths, and
fitted models are never exposed.

---

## Interpretation limits

- The release contains sample IDs but no patient IDs, measurement dates, or diagnosis
  dates. Repeated patients and cohort overlap cannot be verified. Do not report
  patient-disjoint results or months-ahead detection.
- The inspected binary-task files contain 446,663 records; the paper reports 446,558.
  The 105-record discrepancy is retained in the audit.
- The binary files have 76 columns while the paper describes a 72-feature model. This
  is an explicitly different 5/15-feature benchmark, not a reproduction of that model.
- Training files are enriched for the positive label. Ranking, calibration, and
  predictive value must be evaluated separately per cohort.
- The `healthy` label does not independently prove disease absence or describe
  diagnostic verification. Cohort design and diagnoses require further
  study-documentation review before any clinical claim; TRIPOD+AI is a recommended
  reporting reference.
- Longitudinal work requires a patient-linked dataset with a defined outcome timeline;
  mapping feature names alone is insufficient.

---

## License

MIT. Research and non-diagnostic use only. Comply with local regulations and dataset
licenses.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not provide
medical advice, diagnosis, or treatment. Consult qualified healthcare professionals
for medical decisions.

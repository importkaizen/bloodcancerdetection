# Application integration build

## Development checks verification: 2026-09-14

Added the development-only cross-validation runner, aggregate report/API, and [Development checks dashboard](http://127.0.0.1:3000/research?view=development). The actual run completed 50 fold evaluations: four baseline models and a shuffled-training-label logistic control, each with five folds and two measurement sets. All model work used the development cohort. Existing manifest, held-out metrics, and dataset-profile files retained identical SHA-256 hashes before and after this run. Fitted models and record-level predictions were not saved or deployed.

The complete Linux container suite passed **119 tests**, with two upstream Starlette/AnyIO deprecation warnings. Focused Windows CV/API tests passed 37 tests with one host-restricted symlink check skipped. New tests observe actual pipeline fit/predict calls, verify fold-local imputation and scaling, enforce training/evaluation separation, confirm that labels really change in the shuffled control, and prove that modifying non-development features/labels cannot change fold results. The API rejects missing/duplicate folds, wrong cohort/dataset/counts, out-of-range metrics, and feature leakage. Missing CV reports leave other research views usable.

Frontend lint and production builds passed. Live API download matched all 50 saved aggregate results. Browser checks confirmed measurement-set and control selection, per-fold metrics, and the full AUROC chart in the narrow app pane; no console errors or warnings were observed. The shuffled-label control's below-chance means are reported without seed changes or a pass/fail claim. This is an exploratory diagnostic after the initial holdout inspection, not an independent clinical validation study.

## Dataset comparison verification: 2026-09-14

The dataset comparison adds an offline aggregate-profile generator, a read-only profile API, and a dashboard view for all 15 CBC measurements across pooled cohorts and sites. Profiles are tied to a completed benchmark by dataset SHA-256 and cohort counts. The saved profile contains 15 groups and 225 measurement distributions, without individual records. Model fitting, predictions, and holdout performance evaluation were not repeated.

The complete suite passed **105 tests in the Linux API container**, with the same two upstream deprecation warnings. Focused Windows profile/API checks passed 30 tests and skipped the host-restricted symlink check. Tests cover observed-only quantiles, entirely missing measurements, no model fitting/prediction calls, validation-only cohort boundaries, checksum mismatch, refusal to overwrite, API export equivalence, invalid units/quantiles/counts, and missing-profile fallback. Dashboard lint and production build passed.

The live proxied download matched all 225 saved feature distributions. Browser checks exercised cohort and hospital selection, hemoglobin in g/dL, and chart/table layout in the narrow app pane. External hospital hemoglobin medians displayed 11.1, 12.4, 11.9, and 10.5 g/dL for D, E, F, and G. The browser console had no errors or warnings. Zero missingness refers to the prepared release, not verified completeness of source clinical records.

## Research dashboard verification: 2026-09-14

The [research dashboard](http://127.0.0.1:3000/research) now displays the saved eight-experiment benchmark: cohort and measurement-set comparisons, hospital performance, reliability-bin calibration, reproducibility details, and aggregate JSON export. The API and frontend containers were rebuilt locally. No training or holdout evaluation was repeated.

The complete suite passed **90 tests inside the Linux API container**. Windows passed 89 tests and skipped the symbolic-link escape test because the host could not create symbolic links; that check passed in Linux. Two upstream Starlette/AnyIO deprecation warnings remain. Frontend lint and production build passed.

The new API tests compare every returned result with the original saved metrics, check download equivalence, ensure undeclared record data/artifact paths are omitted, reject incomplete or inconsistent reports, and check path containment. Live HTTP checks confirmed the proxied catalog (446,663 records, eight experiments), all 88 aggregate rows, and the JSON attachment response.

Browser checks exercised five/fifteen-measurement selection, XGBoost details, external/test/validation cohorts, chart rendering, and the validation threshold-selection notice. The external expanded-CBC table matched the saved Random Forest AUROC of 0.9643 and XGBoost AUROC of 0.9648; hospital G's XGBoost specificity was 85.91%. The browser console contained no errors or warnings during these checks.

## Container verification completed: 2026-09-14

The isolated `hemasight-check` Compose stack now runs locally at [the dashboard](http://127.0.0.1:3000) and [API documentation](http://127.0.0.1:8000/docs). All seven services started. The Linux container test suite passed **74 tests** (two upstream deprecation warnings). The dashboard production image built successfully, and browser checks confirmed patient navigation, stored model output, and page refresh through Nginx.

Live checks passed for API and proxy health, durable ingestion through RabbitMQ and Celery into PostgreSQL, retry keys, conflicting retries, eight concurrent feature-task replays, and saved risk/anomaly outputs from synthetic model fixtures. A separate fault test stopped RabbitMQ, submitted a request, verified the pending event, restarted RabbitMQ, and confirmed automatic processing without manual consumer restart.

The run found and fixed two container integration problems: the queue consumer exited on an early connection failure, and Nginx retained an obsolete API address after container replacement. The consumer now reconnects, RabbitMQ health checks test listener connectivity, and Nginx uses a shared upstream with dynamic DNS resolution, as supported by [Nginx's upstream module](https://nginx.org/en/docs/http/ngx_http_upstream_module.html#server). Docker dependency installation is cached separately from source changes.

The running database contains synthetic check records only. Worker model versions `synthetic_rf_v2` and `synthetic_if_v2` are disposable integration fixtures, not research-trained or diagnostic models. No LeukoAlert benchmark model was deployed and no holdout was evaluated again.

Reproduce on a fresh isolated stack:

```sh
docker compose -p hemasight-check -f hemasight/docker/docker-compose.yml up --build -d --wait
python scripts/check_container_stack.py --project hemasight-check --fixture-models
python scripts/check_queue_recovery.py --project hemasight-check
```

The fixture flag refuses to overwrite any existing worker artifacts; omit it for subsequent basic stack checks. The recovery script intentionally stops and restarts the broker and restricts project names to the `hemasight-check` prefix. These scripts retain their synthetic database records for inspection. Use `docker compose -p hemasight-check -f hemasight/docker/docker-compose.yml stop` to stop the local stack without deleting its data.

The TimescaleDB extension is installed, but the running application tables are ordinary PostgreSQL tables: the hypertable catalog is empty. Time-series partitioning is not enabled or performance-validated. Broker container recreation/data-loss recovery, exhaustive crash recovery, and optional PyTorch paths remain outside the completed checks.

## Earlier local verification

Local verification on 2026-09-13: **73 Python tests passed**, dashboard lint and production build passed, and the npm audit reported zero vulnerabilities after compatible lockfile updates. Two upstream Starlette/AnyIO deprecation warnings remain in the Python test output. No benchmark models or parameters were changed and no additional holdout evaluation was performed.

## Changes

- Ingestion now saves a blood test and a durable `ingestion_events` row in one transaction. A queue outage leaves that event pending while the API returns 202 for the saved request. The dispatcher retries pending events; publication is marked complete only after RabbitMQ confirms it.
- Clients can supply `Idempotency-Key` on POST requests. An identical retry returns the original blood-test ID; a changed payload using the same key returns 409. Without a key, each request remains a new observation. Blank patient identifiers are rejected.
- Feature, risk, and anomaly tasks reuse existing results when replayed. PostgreSQL parent-row locks serialize competing workers. Feature extraction uses a blood-test ID and feature version; saved scores use a feature ID and model version. These guards do not delete historical duplicates.
- The consumer rejects malformed messages without requeueing them indefinitely and acknowledges valid messages only after task publication. Celery publication confirmations and late acknowledgments are enabled; feature-task broker failures receive bounded retries.
- New Isolation Forest artifacts include training-fitted median imputation and use model version `isolation_forest_v2`. Missing values remain missing until that preprocessing runs. Legacy artifacts reject missing inputs and require retraining. The saved feature version must still match worker feature version `v2`.
- Risk responses include `blood_test_id` and `blood_test_date`. Timeline ordering uses measurement date and stable IDs, while preserving processing timestamps separately. Unlinked historical scores remain in API responses with a null measurement date and are omitted from dashboard timelines.
- Dashboard requests use `/api`, keeping `/patients/:id` available for browser navigation and refresh. Vite and Nginx proxy the prefix to the existing API routes. The container build uses Node 22. Compose uses a named RabbitMQ account for communication between containers.
- Charts show one CBC measurement with its own units, include RBC, preserve date labels across local time zones, and space dates by elapsed time. Missing values remain chart gaps. The page layout has readable contrast, and chart code loads only on the patient route. Fetches are cancelled when navigating away.

## Reproduce local checks

From the repository root, with the Python environment activated:

```sh
python -m pytest -q
```

From `hemasight/frontend/react-dashboard`:

```sh
npm ci
npm run lint
npm run build
npm audit
```

For the application stack on a Docker-equipped machine:

```sh
cd hemasight/docker
docker compose up --build -d
```

The new dispatcher service runs `python -m hemasight.data_pipeline.outbox`. Outside Compose, run that command alongside the existing API, consumer, and Celery worker. `--once` drains one batch of up to 100 pending events and exits. API initialization adds the new table without rewriting the existing tables. Back up an existing database before applying deployment changes.

Example retry-safe request header:

```sh
curl -X POST http://localhost:8000/blood-test \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: research-import-record-001' \
  -d '{"patient_id":"DEMO-001","date":"2024-01-01","wbc":7.2,"rbc":4.8,"platelets":210,"hemoglobin":13.5,"lymphocytes":40}'
```

## What was actually exercised

Integration tests run the real FastAPI lifespan and routes, SQLAlchemy persistence in an isolated SQLite database, worker task bodies, and locally fitted Random Forest/Isolation Forest fixtures. Tests cover queue failure, recovery, partial dispatcher progress, transaction rollback, repeated HTTP requests, replay after partial processing, duplicate worker delivery, missing models, missing first-visit trends, measurement-date ordering, and missing patient responses. Broker boundaries use controlled test doubles.

The real API server and Vite server were also started with an isolated synthetic dataset. Browser checks exercised patient listing, navigation, refreshing a patient URL, measurement switching, and the final dated timeline. The final browser console check showed no errors or warnings. Synthetic display scores were labelled `synthetic_ui_fixture`; they are not model-evaluation results.

## Remaining verification and limitations

Docker Desktop 4.90.0, Docker CLI 29.7.2, Compose 5.5.1, and WSL 2.7.13 were installed locally on 2026-09-13/14. Windows Virtual Machine Platform was enabled successfully; its setup reported no restart needed. `docker compose config --quiet` passed. Docker onboarding was completed before the live verification above.

The initial SQLite tests established sequential replay behavior. The later live checks above additionally exercised PostgreSQL concurrent feature replay and real queue recovery. They do not establish correctness under every crash or delivery interleaving.

Once Docker Desktop is running, use an isolated project for the prepared live checks:

```sh
docker compose -p hemasight-check -f hemasight/docker/docker-compose.yml up --build -d --wait
python scripts/check_container_stack.py --project hemasight-check
```

The stack binds published ports to localhost. The check creates synthetic records, verifies real queue-to-worker feature processing and PostgreSQL concurrent replay, and retains the records for inspection. The optional fixture-model flag also verifies fitted risk/anomaly inference inside the disposable worker container.

Delivery is at least once, not exactly once. Existing deployments may already contain duplicates, and idempotency keys apply only to requests that provide them. Keep the dispatcher running to recover pending events. Permanent processing failures, exhausted task retries, dead-letter monitoring, and automatic reconciliation of missing downstream results still need operational work.

Arrival of an older blood test does not automatically recompute later feature rows. Deliberate cohort reconstruction and model artifact versioning remain necessary for longitudinal research. Do not overwrite a serving model with a newly trained model using the same version and expect replay to replace stored scores. The optional PyTorch routines were not tested in this build.

"""Application integration against an isolated DB and deterministic model fixtures."""
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from hemasight import config
from hemasight.api.main import app
from hemasight.api.routes import blood_test, patients
from hemasight.data_pipeline import outbox
from hemasight.db import models
from hemasight.ml import anomaly, inference, model_training
from hemasight.workers import feature_worker as worker


@pytest.fixture
def application(monkeypatch, tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(models, "get_engine", lambda: engine)
    monkeypatch.setattr(blood_test, "SessionLocal", factory)
    monkeypatch.setattr(patients, "SessionLocal", factory)
    monkeypatch.setattr(worker, "_get_session_factory", lambda: factory)
    publisher = Mock(side_effect=RuntimeError("broker unavailable"))
    monkeypatch.setattr(outbox, "publish_blood_test_ingested", publisher)
    for module in (config, inference, model_training):
        for name, filename in (("RISK_MODEL_PATH", "risk.pkl"), ("SCALER_PATH", "scaler.pkl"),
                               ("MODEL_CONFIG_PATH", "config.json")):
            monkeypatch.setattr(module, name, tmp_path / filename)
    monkeypatch.setattr(model_training, "ML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(anomaly, "ML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(anomaly, "ANOMALY_MODEL_PATH", tmp_path / "anomaly.pkl")
    monkeypatch.setattr(anomaly, "ANOMALY_CONFIG_PATH", tmp_path / "anomaly.json")
    # Execute task bodies synchronously; no real broker or real patient data.
    monkeypatch.setattr(worker.compute_risk_score, "delay", worker.compute_risk_score.run)
    monkeypatch.setattr(worker.compute_anomaly_score, "delay", worker.compute_anomaly_score.run)
    with TestClient(app) as client:
        yield client, factory, publisher, tmp_path
    engine.dispose()


PAYLOAD = {"patient_id": "fixture-patient", "date": "2024-01-01", "wbc": 7.2,
           "rbc": 4.8, "platelets": 210, "hemoglobin": 13.5, "lymphocytes": 40}


def test_startup_ingestion_outage_recovery_and_retry_key(application):
    client, factory, publisher, _ = application
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/blood-test", json=PAYLOAD, headers={"Idempotency-Key": "test-1"})
    assert response.status_code == 202
    blood_id = response.json()["blood_test_id"]
    repeated = client.post("/blood-test", json=PAYLOAD, headers={"Idempotency-Key": "test-1"})
    assert repeated.json() == response.json()
    with factory() as db:
        assert db.query(models.Patient).count() == 1
        assert db.query(models.BloodTest).count() == 1
        assert db.query(models.IngestionEvent).one().published_at is None
    publisher.side_effect = None
    assert outbox.dispatch_pending(session_factory=factory) == 1
    assert outbox.dispatch_pending(session_factory=factory) == 0
    publisher.assert_called_with(blood_id, PAYLOAD["patient_id"])
    with factory() as db:
        assert db.query(models.IngestionEvent).one().published_at is not None
    assert client.get("/patients").json()[0]["external_id"] == PAYLOAD["patient_id"]
    assert client.get("/patients/1/blood-tests").json()[0]["wbc"] == 7.2


def test_conflicting_key_and_invalid_patient_are_rejected(application):
    client, factory, _, _ = application
    headers = {"Idempotency-Key": "same-key"}
    assert client.post("/blood-test", json=PAYLOAD, headers=headers).status_code == 202
    assert client.post("/blood-test", json={**PAYLOAD, "wbc": 10}, headers=headers).status_code == 409
    assert client.post("/blood-test", json={**PAYLOAD, "patient_id": "   "}).status_code == 422
    assert client.post("/blood-test", json=PAYLOAD, headers={"Idempotency-Key": "  "}).status_code == 422
    with factory() as db:
        assert db.query(models.BloodTest).count() == 1


def test_dispatch_batch_preserves_confirmed_progress_on_later_failure(application):
    client, factory, publisher, _ = application
    for day in ("2024-01-01", "2024-01-02"):
        assert client.post("/blood-test", json={**PAYLOAD, "date": day}).status_code == 202
    publisher.side_effect = ["confirmed", RuntimeError("offline")]
    with pytest.raises(RuntimeError, match="offline"):
        outbox.dispatch_pending(session_factory=factory)
    with factory() as db:
        events = db.query(models.IngestionEvent).order_by(models.IngestionEvent.id).all()
        assert events[0].published_at is not None
        assert events[1].published_at is None
    publisher.side_effect = None
    assert outbox.dispatch_pending(session_factory=factory) == 1


def test_replaying_partial_processing_does_not_duplicate_features(application, monkeypatch):
    client, factory, _, _ = application
    test_id = client.post("/blood-test", json=PAYLOAD).json()["blood_test_id"]
    failing_dispatch = Mock(side_effect=RuntimeError("downstream unavailable"))
    monkeypatch.setattr(worker.compute_anomaly_score, "delay", failing_dispatch)
    with pytest.raises(RuntimeError, match="downstream unavailable"):
        worker.process_blood_test.run(test_id)
    failing_dispatch.side_effect = None
    assert worker.process_blood_test.run(test_id)["status"] == "ok"
    with factory() as db:
        assert db.query(models.Feature).count() == 1


def test_outbox_insert_failure_rolls_back_patient_and_test(application):
    client, factory, publisher, _ = application
    def fail(*args):
        raise RuntimeError("private diagnostic details")
    event.listen(models.IngestionEvent, "before_insert", fail)
    try:
        response = client.post("/blood-test", json=PAYLOAD)
    finally:
        event.remove(models.IngestionEvent, "before_insert", fail)
    assert response.status_code == 503
    assert "private diagnostic" not in response.text
    publisher.assert_not_called()
    with factory() as db:
        assert db.query(models.Patient).count() == 0
        assert db.query(models.BloodTest).count() == 0
        assert db.query(models.IngestionEvent).count() == 0


def test_worker_replay_and_saved_models_score_first_visit(application):
    client, factory, _, tmp_path = application
    matrix = np.random.default_rng(42).normal(size=(80, 13))
    frame = pd.DataFrame(matrix, columns=model_training.FEATURE_COLUMNS)
    frame["label"] = np.tile([0, 1], 40)
    path = tmp_path / "synthetic.csv"
    frame.to_csv(path, index=False)
    model_training.train(str(path), model_type="rf", feature_version="v2")
    anomaly.fit_isolation_forest(matrix, feature_version="v2")
    test_id = client.post("/blood-test", json=PAYLOAD).json()["blood_test_id"]
    first = worker.process_blood_test.run(test_id)
    second = worker.process_blood_test.run(test_id)
    assert first == second
    with factory() as db:
        feat = db.query(models.Feature).one()
        assert feat.feature_version == "v2"
        assert feat.wbc_trend is None
        assert db.query(models.RiskScore).count() == 1
        assert db.query(models.AnomalyScore).count() == 1
    result = client.get("/patients/1/risk-scores").json()[0]
    assert result["blood_test_date"].startswith("2024-01-01")
    assert result["blood_test_id"] == test_id
    assert 0 <= result["score"] <= 1


def test_untrained_models_skip_cleanly_and_missing_patient_returns_404(application):
    client, factory, _, _ = application
    test_id = client.post("/blood-test", json=PAYLOAD).json()["blood_test_id"]
    result = worker.process_blood_test.run(test_id)
    assert result["status"] == "ok"
    assert worker.compute_risk_score.run(result["feature_id"])["status"] == "skipped"
    assert worker.compute_anomaly_score.run(result["feature_id"])["status"] == "skipped"
    assert worker.process_blood_test.run(999)["status"] == "not_found"
    assert client.get("/patients/999/blood-tests").status_code == 404
    assert client.get("/patients/999/risk-scores").status_code == 404
    with factory() as db:
        assert db.query(models.RiskScore).count() == 0


def test_timeline_uses_measurement_dates_and_retains_unlinked_scores(application):
    client, factory, _, _ = application
    newer = client.post("/blood-test", json={**PAYLOAD, "date": "2024-02-01"}).json()["blood_test_id"]
    older = client.post("/blood-test", json=PAYLOAD).json()["blood_test_id"]
    with factory() as db:
        for blood_id, computed in ((newer, datetime(2025, 1, 1)), (older, datetime(2025, 2, 1)), (None, datetime(2025, 3, 1))):
            db.add(models.RiskScore(patient_id=1, blood_test_id=blood_id, score=.3,
                                    level="LOW", computed_at=computed, model_version="fixture"))
        db.commit()
    results = client.get("/patients/1/risk-scores").json()
    assert [row["blood_test_id"] for row in results] == [None, older, newer]
    assert results[0]["blood_test_date"] is None
    assert results[-1]["blood_test_date"].startswith("2024-02-01")

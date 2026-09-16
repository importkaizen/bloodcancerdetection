"""Feature-version compatibility for saved Isolation Forest models."""
import json
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import hemasight.db.models as models
from hemasight.ml import anomaly


@pytest.fixture
def anomaly_example(tmp_path, monkeypatch):
    monkeypatch.setattr(anomaly, "ML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(anomaly, "ANOMALY_MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(anomaly, "ANOMALY_CONFIG_PATH", tmp_path / "config.json")
    engine = create_engine("sqlite:///:memory:")
    models.Base.metadata.create_all(engine)
    monkeypatch.setattr(models, "get_engine", lambda: engine)
    # Small deterministic synthetic fixture, never a clinical training dataset.
    matrix = np.random.default_rng(42).normal(size=(30, len(anomaly.FEATURE_COLS)))
    feature_values = dict(zip(anomaly.FEATURE_COLS, matrix[0]))
    with Session(engine) as db:
        db.add(models.Patient(id=1, external_id="anomaly-test"))
        db.add(models.BloodTest(id=1, patient_id=1, date=datetime(2024, 1, 1)))
        db.add_all([
            models.Feature(
                id=1, patient_id=1, blood_test_id=1, feature_version="v1", **feature_values
            ),
            models.Feature(
                id=2, patient_id=1, blood_test_id=1, feature_version="v2", **feature_values
            ),
        ])
        db.commit()
    anomaly.fit_isolation_forest(matrix)
    yield matrix
    engine.dispose()


def assert_scored(feature_id):
    score, is_anomaly = anomaly.compute_anomaly_for_feature_id(feature_id)
    assert np.isfinite(score)
    assert is_anomaly in (0, 1)


def test_default_v1_artifact_accepts_matching_row_and_rejects_v2(anomaly_example):
    config = json.loads(anomaly.ANOMALY_CONFIG_PATH.read_text())
    assert config["feature_version"] == "v1"
    assert_scored(1)
    with pytest.raises(ValueError, match="expects feature version 'v1'.*'v2'.*Retrain"):
        anomaly.compute_anomaly_for_feature_id(2)


@pytest.mark.parametrize("missing_config", [False, True])
def test_legacy_artifacts_are_treated_as_v1(anomaly_example, missing_config):
    if missing_config:
        anomaly.ANOMALY_CONFIG_PATH.unlink()
    else:
        config = json.loads(anomaly.ANOMALY_CONFIG_PATH.read_text())
        del config["feature_version"]
        anomaly.ANOMALY_CONFIG_PATH.write_text(json.dumps(config))
    assert_scored(1)
    with pytest.raises(ValueError, match="expects feature version 'v1'.*'v2'.*Retrain"):
        anomaly.compute_anomaly_for_feature_id(2)


def test_retrained_v2_artifact_accepts_v2_and_rejects_v1(anomaly_example):
    anomaly.fit_isolation_forest(anomaly_example, feature_version="v2")
    config = json.loads(anomaly.ANOMALY_CONFIG_PATH.read_text())
    assert config["feature_version"] == "v2"
    assert_scored(2)
    with pytest.raises(ValueError, match="expects feature version 'v2'.*'v1'.*Retrain"):
        anomaly.compute_anomaly_for_feature_id(1)

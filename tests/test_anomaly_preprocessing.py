import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from hemasight.ml import anomaly


def test_anomaly_model_reuses_training_medians(tmp_path, monkeypatch):
    monkeypatch.setattr(anomaly, "ML_MODELS_DIR", tmp_path)
    monkeypatch.setattr(anomaly, "ANOMALY_MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(anomaly, "ANOMALY_CONFIG_PATH", tmp_path / "config.json")
    matrix = np.random.default_rng(42).normal(size=(40, 13))
    matrix[:5, 0] = np.nan
    anomaly.fit_isolation_forest(matrix, feature_version="v2")
    model = joblib.load(anomaly.ANOMALY_MODEL_PATH)
    np.testing.assert_allclose(model.named_steps["imputer"].statistics_, np.nanmedian(matrix, axis=0))
    vector = anomaly.feature_row_to_vector({})
    assert np.isnan(vector).all()
    score, flag = anomaly.predict_anomaly(vector)
    median_vector = np.nanmedian(matrix, axis=0).reshape(1, -1)
    assert score == pytest.approx(-model.decision_function(median_vector)[0])
    assert flag in (0, 1)
    assert anomaly.artifact_version() == "isolation_forest_v2"


def test_legacy_anomaly_model_rejects_missing_values():
    model = IsolationForest(random_state=42).fit(np.random.default_rng(42).normal(size=(20, 13)))
    assert np.isfinite(anomaly.predict_anomaly(np.ones((1, 13)), model)[0])
    with pytest.raises(ValueError, match="Legacy anomaly"):
        anomaly.predict_anomaly(np.full((1, 13), np.nan), model)


def test_anomaly_training_rejects_unusable_columns():
    with pytest.raises(ValueError, match="entirely missing"):
        anomaly.fit_isolation_forest(np.full((20, 13), np.nan))

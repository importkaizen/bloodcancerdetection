"""Regression checks for train-only preprocessing and saved-model inference."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from hemasight.ml import inference, model_training as training


@pytest.fixture
def training_example(tmp_path, monkeypatch):
    for module in (training, inference):
        for name, filename in (("RISK_MODEL_PATH", "model.pkl"),
                               ("SCALER_PATH", "scaler.pkl"),
                               ("MODEL_CONFIG_PATH", "config.json")):
            monkeypatch.setattr(module, name, tmp_path / filename)
    monkeypatch.setattr(training, "ML_MODELS_DIR", tmp_path)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(80, len(training.FEATURE_COLUMNS))),
                         columns=training.FEATURE_COLUMNS)
    frame["label"] = np.tile([0, 1], 40)
    frame["patient"] = np.repeat(np.arange(20), 4)
    path = tmp_path / "fixture.csv"
    return frame, path


def test_saved_preprocessing_uses_training_rows_and_inference_reuses_it(training_example):
    frame, path = training_example
    train_idx, test_idx, _ = training._split_indices(frame.label, .2, 42)
    frame.loc[train_idx[:10], "wbc"] = np.nan
    frame.loc[test_idx, "wbc"] = 10000
    frame.to_csv(path, index=False)
    training.train(str(path), model_type="rf")
    prep = joblib.load(training.SCALER_PATH)
    expected = frame.iloc[train_idx][training.FEATURE_COLUMNS].median().to_numpy()
    np.testing.assert_allclose(prep.named_steps["imputer"].statistics_, expected)
    imputed_train = frame.iloc[train_idx][training.FEATURE_COLUMNS].fillna(
        dict(zip(training.FEATURE_COLUMNS, expected)))
    np.testing.assert_allclose(prep.named_steps["scaler"].mean_, imputed_train.mean())
    row = frame.iloc[0][training.FEATURE_COLUMNS].to_dict()
    row["wbc"] = None
    vector = inference.feature_row_to_vector(row)
    assert np.isnan(vector[0, 0])
    result = inference.compute_risk(vector)
    model = joblib.load(training.RISK_MODEL_PATH)
    expected_score = model.predict_proba(prep.transform(
        pd.DataFrame(vector, columns=training.FEATURE_COLUMNS)))[0, 1]
    assert result["score"] == round(float(expected_score), 4)
    assert result["model_version"] == "v2"


def test_patient_groups_are_disjoint_and_saved_in_metadata(training_example):
    frame, path = training_example
    train_idx, test_idx, details = training._split_indices(frame.label, .2, 42, frame.patient)
    assert set(frame.patient.iloc[train_idx]).isdisjoint(frame.patient.iloc[test_idx])
    frame.to_csv(path, index=False)
    result = training.train(str(path), model_type="rf", patient_group_col="patient")
    config = json.loads(training.MODEL_CONFIG_PATH.read_text())
    assert result["run"]["split"] == details == config["run"]["split"]
    assert config["run"]["train_rows"] + config["run"]["test_rows"] == len(frame)


@pytest.mark.parametrize("missing", [None, "  "])
def test_missing_patient_identifier_rejected(training_example, missing):
    frame, _ = training_example
    groups = frame.patient.astype(object)
    groups.iloc[0] = missing
    with pytest.raises(ValueError, match="missing identifiers"):
        training._split_indices(frame.label, .2, 42, groups)


def test_legacy_scaler_rejects_missing_features_but_accepts_complete(training_example):
    frame, path = training_example
    frame.to_csv(path, index=False)
    training.train(str(path), model_type="rf")
    model = joblib.load(training.RISK_MODEL_PATH)
    config = json.loads(training.MODEL_CONFIG_PATH.read_text())
    scaler = StandardScaler().fit(frame[training.FEATURE_COLUMNS].to_numpy())
    vector = frame.iloc[[0]][training.FEATURE_COLUMNS].to_numpy(copy=True)
    assert 0 <= inference.compute_risk(vector, model, scaler, config)["score"] <= 1
    vector[0, 0] = np.nan
    with pytest.raises(ValueError, match="legacy artifact"):
        inference.compute_risk(vector, model, scaler, config)


def test_version_mismatch_and_all_missing_training_feature_do_not_save(training_example):
    frame, path = training_example
    frame["feature_version"] = "v2"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="declared feature_version"):
        training.train(str(path), model_type="rf")
    frame["wbc"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="entirely missing"):
        training.train(str(path), model_type="rf", feature_version="v2")
    assert not training.RISK_MODEL_PATH.exists()


def test_vector_uses_saved_column_order_and_rejects_infinity():
    np.testing.assert_equal(inference.feature_row_to_vector(
        {"wbc": 3, "rbc": 5}, ["rbc", "wbc"]), [[5, 3]])
    with pytest.raises(ValueError, match="infinite"):
        inference.feature_row_to_vector({"wbc": np.inf}, ["wbc"])

"""Cross-validation leakage boundaries and deterministic fold diagnostics."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from hemasight.research import cross_validation as cv
from hemasight.research.schema import FEATURE_SETS


@pytest.fixture
def prepared_cv(tmp_path):
    frame = pd.DataFrame({"sample_id": [f"PRIVATE_{i}" for i in range(24)],
                          "cohort": ["development"] * 16 + ["validation"] * 4 + ["external"] * 4,
                          "site": ["A"] * 20 + ["D"] * 4, "target": [0, 1] * 12})
    for i, name in enumerate(FEATURE_SETS["basic_cbc"]):
        frame[name] = np.arange(24, dtype=float) + i
        frame.loc[2, name] = np.nan
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    run = tmp_path / "run"
    run.mkdir()
    manifest = {"status": "completed", "dataset": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                "models": ["prevalence", "logistic"], "feature_sets": {"basic_cbc": FEATURE_SETS["basic_cbc"]},
                "cohort_counts": {"development": {"n": 16, "n_positive": 8}}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    return path, run, frame


def test_preprocessing_and_predictions_stay_inside_development_folds(prepared_cv, monkeypatch):
    dataset, run, frame = prepared_cv
    factory = cv.build_pipeline
    evaluations = []
    fitted_labels = []
    def instrumented(name, seed):
        pipeline = factory(name, seed)
        fit, predict = pipeline.fit, pipeline.predict_proba
        trained = set()
        def checked_fit(X, y):
            trained.update(X.index)
            fitted_labels.append(np.asarray(y).copy())
            assert trained <= set(range(16))
            result = fit(X, y)
            np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, X.median().to_numpy())
            if name == "logistic":
                np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, X.fillna(X.median()).mean().to_numpy())
            return result
        def checked_predict(X):
            assert set(X.index) <= set(range(16))
            assert not trained.intersection(X.index)
            evaluations.append(set(X.index))
            return predict(X)
        pipeline.fit, pipeline.predict_proba = checked_fit, checked_predict
        return pipeline
    monkeypatch.setattr(cv, "build_pipeline", instrumented)
    report = cv.run_cross_validation(dataset, run, n_splits=4)
    assert len(report["results"]) == 12
    assert set.union(*evaluations) == set(range(16))
    assert len(evaluations) == 12
    assert any(not np.array_equal(fitted_labels[4+i], fitted_labels[8+i]) for i in range(4))
    assert "PRIVATE" not in (run / "cross_validation.json").read_text()
    controls = [r for r in report["results"] if r["control"] != "original"]
    assert len(controls) == 4
    assert all(r["train_positive"] == 6 and r["evaluation_positive"] == 2 for r in controls)


def test_other_cohorts_cannot_change_fold_results(prepared_cv):
    dataset, run, frame = prepared_cv
    first = cv.run_cross_validation(dataset, run, n_splits=4)
    second_run = run.parent / "changed-holdouts"
    second_run.mkdir()
    frame.loc[frame.cohort != "development", FEATURE_SETS["basic_cbc"]] = 99999.0
    frame.loc[frame.cohort != "development", "target"] = 1 - frame.loc[frame.cohort != "development", "target"]
    frame.to_csv(dataset, index=False)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["dataset"]["sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    (second_run / "manifest.json").write_text(json.dumps(manifest))
    second = cv.run_cross_validation(dataset, second_run, n_splits=4)
    assert first["results"] == second["results"]


def test_existing_report_and_insufficient_classes_fail_closed(prepared_cv):
    dataset, run, _ = prepared_cv
    with pytest.raises(ValueError, match="at least n_splits"):
        cv.run_cross_validation(dataset, run, n_splits=10)
    output = run / "cross_validation.json"
    assert not output.exists()
    output.write_text("preserve me")
    with pytest.raises(FileExistsError):
        cv.run_cross_validation(dataset, run)
    assert output.read_text() == "preserve me"


def test_dataset_mismatch_prevents_training(prepared_cv, monkeypatch):
    dataset, run, _ = prepared_cv
    dataset.write_text(dataset.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        cv.run_cross_validation(dataset, run)
    assert not (run / "cross_validation.json").exists()

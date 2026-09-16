"""Regression checks for leakage, cohort boundaries, and durable research outputs."""

from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from hemasight.research import benchmark
from hemasight.research.schema import FEATURE_SETS


@pytest.fixture
def dataset(tmp_path):
    cohorts = ["development"] * 8 + ["validation"] * 6 + ["external"] * 4 + ["test"] * 4
    frame = pd.DataFrame({
        "sample_id": [f"PRIVATE_RECORD_{i}" for i in range(len(cohorts))],
        "site": ["A"] * 8 + ["A"] * 3 + ["B"] * 3 + ["D"] * 2 + ["E"] * 2 + ["real_world"] * 4,
        "cohort": cohorts,
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
    })
    for index, name in enumerate(FEATURE_SETS["expanded_cbc"]):
        frame[name] = np.array([1, 2, np.nan, 3, 6, 7, 8, 9] + list(range(10, 24)), dtype=float) + index
    # A decoy perfectly predicts the label but must never reach the estimator.
    frame["diagnosis_leak"] = frame.target
    path = tmp_path / "features.csv"
    frame.to_csv(path, index=False)
    return path, frame


def run_small(dataset, tmp_path, name="run", **kwargs):
    path, _ = dataset
    return benchmark.run_benchmark(
        path, tmp_path / name, models=["logistic"], feature_sets=["basic_cbc"], **kwargs
    )


def test_imputation_and_scaling_fit_only_development(dataset, tmp_path):
    path, frame = dataset
    frame.loc[frame.cohort != "development", "wbc"] = 10000.0
    frame.to_csv(path, index=False)
    run_small(dataset, tmp_path)
    bundle = joblib.load(tmp_path / "run/models/basic_cbc__logistic.joblib")
    pipeline = bundle["pipeline"]
    expected_median = frame.loc[frame.cohort == "development", "wbc"].median()
    assert pipeline.named_steps["imputer"].statistics_[0] == expected_median == 6.0
    expected_mean = frame.loc[frame.cohort == "development", "wbc"].fillna(expected_median).mean()
    assert pipeline.named_steps["scaler"].mean_[0] == expected_mean
    assert list(pipeline.feature_names_in_) == FEATURE_SETS["basic_cbc"]
    assert "diagnosis_leak" not in bundle["feature_columns"]
    assert np.isfinite(pipeline.predict_proba(frame.loc[:, FEATURE_SETS["basic_cbc"]])).all()


def test_default_does_not_predict_holdouts(dataset, tmp_path, monkeypatch):
    _, frame = dataset
    predicted_indices = []
    original = benchmark.Pipeline.predict_proba

    def tracked(self, X, *args, **kwargs):
        predicted_indices.extend(X.index.tolist())
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(benchmark.Pipeline, "predict_proba", tracked)
    result = run_small(dataset, tmp_path)
    assert set(predicted_indices) == set(frame.index[frame.cohort == "validation"])
    assert {row["cohort"] for row in result["results"]} == {"validation"}
    assert result["manifest"]["evaluated_cohorts"] == ["validation"]


def test_holdout_changes_cannot_change_fit_or_threshold(dataset, tmp_path):
    path, frame = dataset
    first = run_small(dataset, tmp_path, "before", evaluate_holdouts=True)
    heldout = frame.cohort.isin(["external", "test"])
    frame.loc[heldout, "target"] = 1 - frame.loc[heldout, "target"]
    frame.loc[heldout, FEATURE_SETS["basic_cbc"]] = 1_000_000.0
    frame.to_csv(path, index=False)
    second = run_small(dataset, tmp_path, "after", evaluate_holdouts=True)
    assert [row["threshold"] for row in first["manifest"]["experiments"]] == [
        row["threshold"] for row in second["manifest"]["experiments"]
    ]
    model_before = joblib.load(tmp_path / "before/models/basic_cbc__logistic.joblib")["pipeline"]
    model_after = joblib.load(tmp_path / "after/models/basic_cbc__logistic.joblib")["pipeline"]
    np.testing.assert_array_equal(model_before[-1].coef_, model_after[-1].coef_)
    assert {row["cohort"] for row in first["results"]} == {"validation", "external", "test"}


def test_threshold_ties_achieve_specificity_and_maximum_sensitivity():
    labels = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.3, 0.3, 0.9, 0.3, 0.31, 0.95])
    threshold = benchmark.threshold_at_specificity(labels, scores, 0.75)
    metrics = benchmark.classification_metrics(labels, scores, threshold)
    assert metrics["specificity"] == 0.75
    assert metrics["sensitivity"] == 2 / 3
    assert threshold > 0.3
    assert benchmark.classification_metrics(labels, scores, 0.3)["specificity"] < 0.75
    threshold = benchmark.threshold_at_specificity([0, 0, 1], [1.0, 1.0, 1.0], 0.95)
    assert threshold > 1.0
    assert benchmark.classification_metrics([0, 0, 1], [1.0, 1.0, 1.0], threshold)["specificity"] == 1.0


def test_site_counts_sum_to_overall_and_single_class_is_defined_safely(dataset, tmp_path):
    result = run_small(dataset, tmp_path, evaluate_holdouts=True)
    for cohort in ("validation", "external", "test"):
        rows = [row for row in result["results"] if row["cohort"] == cohort and row["model"] == "logistic"]
        overall = next(row["metrics"] for row in rows if row["scope"] == "overall")
        sites = [row["metrics"] for row in rows if row["scope"] == "site"]
        for key in ("n", "n_positive", "n_negative", "true_positives", "true_negatives", "false_positives", "false_negatives"):
            assert sum(site[key] for site in sites) == overall[key]
    negative_site = next(row for row in result["results"] if row["site"] == "D")
    assert negative_site["metrics"]["auroc"] is None
    assert negative_site["metrics"]["average_precision"] is None
    assert negative_site["metrics"]["sensitivity"] is None
    positive_site = next(row for row in result["results"] if row["site"] == "E")
    assert positive_site["metrics"]["specificity"] is None
    assert positive_site["metrics"]["false_positives_per_1000_negatives"] is None
    assert sum(b["n"] for b in negative_site["reliability_bins"]) == negative_site["metrics"]["n"]


def test_reliability_bins_include_one_and_empty_bins():
    bins = benchmark.reliability_bins([0, 1, 1], [0.0, 0.9, 1.0])
    assert sum(item["n"] for item in bins) == 3
    assert bins[-1]["n"] == 2
    assert bins[-1]["observed_positive_fraction"] == 1
    assert bins[1]["mean_predicted_probability"] is None


def test_outputs_have_provenance_no_record_identifiers_and_cannot_overwrite(dataset, tmp_path):
    path, _ = dataset
    result = run_small(dataset, tmp_path)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["dataset"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["source_code"]["files_sha256"]["benchmark.py"]
    assert "scikit-learn" in manifest["versions"]
    assert len(result["manifest"]["experiments"]) == 2
    for file in (tmp_path / "run").glob("*.*"):
        assert "PRIVATE_RECORD_" not in file.read_text(encoding="utf-8")
    original = (tmp_path / "run/manifest.json").read_bytes()
    with pytest.raises(ValueError, match="never overwritten"):
        run_small(dataset, tmp_path)
    assert (tmp_path / "run/manifest.json").read_bytes() == original


def test_preparation_manifest_integrity_and_selected_provenance(dataset, tmp_path):
    path, _ = dataset
    source_manifest = {
        "prepared_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_archive_sha256": "a" * 64,
        "source_doi": "10.17632/vc7kwnyppz.1",
        "record_data": "PRIVATE_RECORD_0",
    }
    manifest_path = path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    result = run_small(dataset, tmp_path)
    provenance = result["manifest"]["preparation_provenance"]
    assert provenance["source_archive_sha256"] == "a" * 64
    assert "record_data" not in provenance
    source_manifest["prepared_csv_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        run_small(dataset, tmp_path, "bad-provenance")
    assert not (tmp_path / "bad-provenance").exists()


@pytest.mark.parametrize("change, error", [
    ("missing_column", "missing required columns"),
    ("target_missing", "binary"),
    ("target_nonbinary", "binary"),
    ("cohort_invalid", "Unknown cohort"),
    ("one_class_development", "development must contain both"),
    ("one_class_validation", "validation must contain both"),
    ("empty_feature", "entirely missing"),
    ("infinite_feature", "infinite"),
    ("nonnumeric_feature", "nonnumeric"),
    ("site_missing", "site values"),
])
def test_invalid_datasets_fail_before_artifact_creation(dataset, tmp_path, change, error):
    path, frame = dataset
    if change == "missing_column":
        frame = frame.drop(columns="wbc")
    elif change == "target_missing":
        frame["target"] = frame.target.astype(float)
        frame.loc[0, "target"] = np.nan
    elif change == "target_nonbinary":
        frame.loc[0, "target"] = 2
    elif change == "cohort_invalid":
        frame.loc[0, "cohort"] = "random_split"
    elif change == "one_class_development":
        frame.loc[frame.cohort == "development", "target"] = 0
    elif change == "one_class_validation":
        frame.loc[frame.cohort == "validation", "target"] = 1
    elif change == "empty_feature":
        frame.loc[frame.cohort == "development", "wbc"] = np.nan
    elif change == "infinite_feature":
        frame.loc[0, "wbc"] = np.inf
    elif change == "nonnumeric_feature":
        frame["wbc"] = frame.wbc.astype(object)
        frame.loc[0, "wbc"] = "unknown"
    elif change == "site_missing":
        frame.loc[0, "site"] = ""
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        run_small(dataset, tmp_path)
    assert not (tmp_path / "run").exists()


def test_forbidden_metadata_cannot_enter_schema(dataset, tmp_path, monkeypatch):
    monkeypatch.setitem(benchmark.FEATURE_SETS, "basic_cbc", ["wbc", "target"])
    with pytest.raises(ValueError, match="forbidden metadata"):
        run_small(dataset, tmp_path)


def test_holdout_flag_fails_if_requested_cohorts_absent(dataset, tmp_path):
    path, frame = dataset
    frame.loc[frame.cohort != "test"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="cohorts are absent"):
        run_small(dataset, tmp_path, evaluate_holdouts=True)


def test_all_requested_estimators_and_feature_sets_complete(dataset, tmp_path):
    path, _ = dataset
    result = benchmark.run_benchmark(path, tmp_path / "all", models=["logistic", "rf"])
    assert len(result["manifest"]["experiments"]) == 6
    for item in result["manifest"]["experiments"]:
        assert (tmp_path / "all" / item["model_bundle"]).is_file()


def test_xgboost_optional_estimator_handles_missing_features(dataset, tmp_path):
    pytest.importorskip("xgboost")
    path, _ = dataset
    result = benchmark.run_benchmark(
        path, tmp_path / "xgb", models=["xgboost"], feature_sets=["basic_cbc"]
    )
    assert result["manifest"]["experiments"][-1]["model"] == "xgboost"


def test_cli_is_runnable(dataset, tmp_path, capsys):
    path, _ = dataset
    assert benchmark.main([
        str(path), "--output", str(tmp_path / "cli"), "--models", "logistic",
        "--feature-sets", "basic_cbc", "--target-specificity", "0.9",
    ]) == 0
    assert "Evaluated cohorts: validation" in capsys.readouterr().out

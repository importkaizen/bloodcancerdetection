"""Distribution arithmetic, cohort boundaries, and immutable profile generation."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from hemasight.research.profile import build_profile, summarize_group
from hemasight.research.profile_schema import DatasetProfile
from hemasight.research.schema import COLUMN_MAP


@pytest.fixture
def prepared(tmp_path):
    frame = pd.DataFrame({"sample_id": [f"PRIVATE_{i}" for i in range(12)],
                          "cohort": ["development"] * 4 + ["validation"] * 4 + ["external"] * 4,
                          "site": ["A"] * 4 + ["B"] * 4 + ["D"] * 4,
                          "target": [0, 1] * 6})
    for feature in COLUMN_MAP:
        frame[feature] = [1., 2., 3., 4., 5., np.nan, 7., 8., 10., 20., 30., 40.]
    frame.loc[frame.cohort == "external", "wbc"] = np.nan
    dataset = tmp_path / "features.csv"
    frame.to_csv(dataset, index=False)
    manifest = {"status": "completed", "dataset": {"sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "rows": len(frame)},
                "evaluated_cohorts": ["validation", "external"], "evaluate_holdouts": True,
                "cohort_counts": {c: {"n": len(g), "n_positive": int(g.target.sum())} for c, g in frame.groupby("cohort")}}
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset, run, frame


def test_quantiles_use_only_observed_values(prepared):
    _, _, frame = prepared
    summary = summarize_group(frame.loc[frame.cohort == "validation"], "validation")
    values = summary["features"]["wbc"]
    assert values == {"observed": 3, "missing": 1, "p05": 5.2, "q1": 6.0, "median": 7.0, "q3": 7.5, "p95": 7.9}


def test_profile_preserves_missingness_without_model_calls(prepared, monkeypatch):
    dataset, run, _ = prepared
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset profiling must not fit or predict")
    monkeypatch.setattr(Pipeline, "fit", forbidden)
    monkeypatch.setattr(Pipeline, "predict_proba", forbidden)
    path = build_profile(dataset, run)
    report = DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
    empty = next(g for g in report.groups if g.cohort == "external" and g.scope == "overall").features["wbc"]
    assert empty.observed == 0 and empty.missing == 4 and empty.median is None
    assert "PRIVATE" not in path.read_text(encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        build_profile(dataset, run)
    assert path.read_bytes() == before


def test_validation_only_run_does_not_summarize_holdouts(prepared):
    dataset, run, _ = prepared
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(evaluated_cohorts=["validation"], evaluate_holdouts=False)
    path.write_text(json.dumps(manifest))
    profile = json.loads(build_profile(dataset, run).read_text())
    assert {g["cohort"] for g in profile["groups"]} == {"development", "validation"}


@pytest.mark.parametrize("change", [
    lambda d: d["dataset"].update(sha256="0" * 64),
    lambda d: d.update(evaluate_holdouts=False),
    lambda d: d.update(status="running"),
    lambda d: d["cohort_counts"]["external"].update(n=100),
])
def test_invalid_run_cannot_generate_profile(prepared, change):
    dataset, run, _ = prepared
    path = run / "manifest.json"
    manifest = json.loads(path.read_text())
    change(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        build_profile(dataset, run)
    assert not (run / "profile.json").exists()

"""Exercise saved-report integrity and the public aggregate-only API boundary."""
import json
from pathlib import Path
import shutil

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from hemasight.api.routes import research

SOURCE = Path(__file__).resolve().parents[1] / "docs/results/holdout-v1"


@pytest.fixture
def reports(tmp_path, monkeypatch):
    directory = tmp_path / "holdout-v1"
    shutil.copytree(SOURCE, directory)
    monkeypatch.setattr(research, "RESEARCH_REPORTS_DIR", tmp_path)
    app = FastAPI()
    app.include_router(research.router)
    with TestClient(app) as client:
        yield client, directory


def rewrite(directory, filename, change):
    path = directory / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_saved_results_and_download_match_original_metrics(reports):
    client, directory = reports
    catalog = client.get("/research/runs").json()
    assert catalog["unavailable_runs"] == 0
    assert catalog["runs"][0]["records"] == 446663
    assert catalog["runs"][0]["experiments"] == 8
    response = client.get("/research/runs/holdout-v1")
    assert response.status_code == 200
    report = response.json()
    original = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert report["results"] == original["results"]
    download = client.get("/research/runs/holdout-v1?download=true")
    assert download.json() == report
    assert download.headers["content-disposition"] == 'attachment; filename="holdout-v1-aggregate-results.json"'


def test_unknown_record_data_and_artifact_paths_are_not_exposed(reports):
    client, directory = reports
    rewrite(directory, "manifest.json", lambda d: d.update(patient_records=[{"secret": "private"}], model_path="private/model.pkl"))
    rewrite(directory, "metrics.json", lambda d: d["results"][0].update(patient_id="private"))
    response = client.get("/research/runs/holdout-v1?download=true")
    assert response.status_code == 200
    assert "private" not in response.text
    assert "model_bundle" not in response.text
    assert "filename" not in response.json()["manifest"]["dataset"]


@pytest.mark.parametrize("filename,change", [
    ("manifest.json", lambda d: d.update(status="running")),
    ("manifest.json", lambda d: d.update(models=[])),
    ("manifest.json", lambda d: d["cohort_counts"].pop("development")),
    ("manifest.json", lambda d: d["dataset"].update(rows=1)),
    ("manifest.json", lambda d: d.update(evaluate_holdouts=False)),
    ("metrics.json", lambda d: d["results"].pop(0)),
    ("metrics.json", lambda d: d["results"].append(d["results"][0])),
    ("metrics.json", lambda d: d["results"][0]["metrics"].update(n=1)),
    ("metrics.json", lambda d: d["results"][0]["metrics"].update(specificity=0.123)),
    ("metrics.json", lambda d: d["results"][0]["metrics"].update(threshold=99)),
    ("metrics.json", lambda d: d["results"][0].update(reliability_bins=[])),
    ("metrics.json", lambda d: d.update(results=[r for r in d["results"] if r["scope"] == "overall"])),
])
def test_inconsistent_reports_are_unavailable(reports, filename, change):
    client, directory = reports
    rewrite(directory, filename, change)
    response = client.get("/research/runs/holdout-v1")
    assert response.status_code == 503
    assert response.json() == {"detail": "Research report is incomplete or invalid"}
    assert client.get("/research/runs").json() == {"runs": [], "unavailable_runs": 1}


def test_empty_catalog_missing_run_and_invalid_json(reports):
    client, directory = reports
    assert client.get("/research/runs/unknown").status_code == 404
    assert client.get("/research/runs/%2E%2E%5Csecret").status_code == 404
    (directory / "metrics.json").write_text("broken JSON", encoding="utf-8")
    assert client.get("/research/runs/holdout-v1").status_code == 503
    shutil.rmtree(directory)
    assert client.get("/research/runs").json() == {"runs": [], "unavailable_runs": 0}


def test_report_directory_cannot_escape_configured_root(reports, tmp_path):
    client, directory = reports
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    shutil.copytree(directory, outside)
    link = tmp_path / "escaped"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this host")
    assert client.get("/research/runs/escaped").status_code == 404


def test_profile_download_matches_saved_aggregates(reports):
    client, directory = reports
    response = client.get("/research/runs/holdout-v1/profile")
    assert response.status_code == 200
    assert response.json() == json.loads((directory / "profile.json").read_text(encoding="utf-8"))
    download = client.get("/research/runs/holdout-v1/profile?download=true")
    assert download.json() == response.json()
    assert download.headers["content-disposition"] == 'attachment; filename="holdout-v1-dataset-profile.json"'
    rewrite(directory, "profile.json", lambda d: d.update(patient_records=["PRIVATE_RECORD"]))
    assert "PRIVATE_RECORD" not in client.get("/research/runs/holdout-v1/profile").text


@pytest.mark.parametrize("change", [
    lambda d: d.update(dataset_sha256="0" * 64),
    lambda d: d["groups"][0]["features"]["wbc"].update(observed=1),
    lambda d: d["groups"][0]["features"]["wbc"].update(median=-100),
    lambda d: d["units"].update(hemoglobin="g/L"),
    lambda d: d["groups"].pop(0),
    lambda d: d["groups"].append(d["groups"][0]),
])
def test_corrupted_profile_is_not_served(reports, change):
    client, directory = reports
    rewrite(directory, "profile.json", change)
    response = client.get("/research/runs/holdout-v1/profile")
    assert response.status_code == 503
    assert response.json() == {"detail": "Dataset profile is incomplete or invalid"}


def test_absent_profile_does_not_break_model_report(reports):
    client, directory = reports
    (directory / "profile.json").unlink()
    assert client.get("/research/runs/holdout-v1/profile").status_code == 404
    assert client.get("/research/runs/holdout-v1").status_code == 200


def test_development_checks_match_saved_report_and_export(reports):
    client, directory = reports
    response = client.get("/research/runs/holdout-v1/cross-validation")
    assert response.status_code == 200
    assert response.json() == json.loads((directory / "cross_validation.json").read_text(encoding="utf-8"))
    assert len(response.json()["results"]) == 50
    download = client.get("/research/runs/holdout-v1/cross-validation?download=true")
    assert download.json() == response.json()
    assert download.headers["content-disposition"] == 'attachment; filename="holdout-v1-development-checks.json"'
    rewrite(directory, "cross_validation.json", lambda d: d.update(record_predictions=["PRIVATE"]))
    assert "PRIVATE" not in client.get("/research/runs/holdout-v1/cross-validation").text


@pytest.mark.parametrize("change", [
    lambda d: d.update(dataset_sha256="0" * 64),
    lambda d: d.update(evaluation_cohort="external"),
    lambda d: d["results"].pop(),
    lambda d: d["results"].append(d["results"][0]),
    lambda d: d["results"][0].update(train_n=1),
    lambda d: d["results"][0].update(evaluation_positive=1),
    lambda d: d["results"][0].update(auroc=1.5),
    lambda d: d["feature_sets"]["basic_cbc"].append("target"),
])
def test_invalid_development_checks_are_rejected(reports, change):
    client, directory = reports
    rewrite(directory, "cross_validation.json", change)
    response = client.get("/research/runs/holdout-v1/cross-validation")
    assert response.status_code == 503
    assert response.json() == {"detail": "Development checks are incomplete or invalid"}


def test_absent_development_checks_leave_other_views_usable(reports):
    client, directory = reports
    (directory / "cross_validation.json").unlink()
    assert client.get("/research/runs/holdout-v1/cross-validation").status_code == 404
    assert client.get("/research/runs/holdout-v1").status_code == 200
    assert client.get("/research/runs/holdout-v1/profile").status_code == 200

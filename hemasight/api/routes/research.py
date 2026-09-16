"""Read-only access to completed aggregate benchmark reports."""
import json
import math
from pathlib import Path
import re
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, model_validator

from hemasight.config import RESEARCH_REPORTS_DIR
from hemasight.research.profile_schema import DatasetProfile
from hemasight.research.cv_schema import CrossValidationReport

router = APIRouter(prefix="/research/runs", tags=["research"])
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Cohort = Literal["development", "validation", "external", "test"]
Model = Literal["prevalence", "logistic", "rf", "xgboost"]
FeatureSet = Literal["basic_cbc", "expanded_cbc"]


class Counts(BaseModel):
    n: int = Field(ge=0)
    n_positive: int = Field(ge=0)
    n_negative: int = Field(ge=0)

    @model_validator(mode="after")
    def consistent_counts(self):
        if self.n != self.n_positive + self.n_negative:
            raise ValueError("Cohort counts do not add up")
        return self


class Metrics(Counts):
    prevalence: Probability
    auroc: Probability | None
    average_precision: Probability | None
    brier: Probability
    sensitivity: Probability | None
    specificity: Probability | None
    precision: Probability | None
    false_positives_per_1000_negatives: Annotated[float, Field(ge=0, le=1000, allow_inf_nan=False)] | None
    true_positives: int = Field(ge=0)
    true_negatives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    threshold: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def consistent_confusion(self):
        if self.true_positives + self.false_negatives != self.n_positive:
            raise ValueError("Positive counts do not add up")
        if self.true_negatives + self.false_positives != self.n_negative:
            raise ValueError("Negative counts do not add up")
        for field, numerator, denominator in (
            ("prevalence", self.n_positive, self.n),
            ("sensitivity", self.true_positives, self.n_positive),
            ("specificity", self.true_negatives, self.n_negative),
            ("precision", self.true_positives, self.true_positives + self.false_positives),
            ("false_positives_per_1000_negatives", 1000 * self.false_positives, self.n_negative),
        ):
            value = getattr(self, field)
            if denominator == 0:
                if value is not None:
                    raise ValueError("Undefined metric must be null")
            elif value is None or not math.isclose(value, numerator / denominator, abs_tol=1e-10):
                raise ValueError("Metric disagrees with counts")
        return self


class ReliabilityBin(BaseModel):
    lower: Probability
    upper: Probability
    upper_inclusive: bool
    n: int = Field(ge=0)
    mean_predicted_probability: Probability | None
    observed_positive_fraction: Probability | None


class Result(BaseModel):
    feature_set: FeatureSet
    model: Model
    cohort: Literal["validation", "external", "test"]
    scope: Literal["overall", "site"]
    site: str | None = Field(max_length=80)
    metrics: Metrics
    reliability_bins: list[ReliabilityBin]


class Dataset(BaseModel):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(ge=1)


class Provenance(BaseModel):
    source_doi: str | None = None
    source_license: str | None = None
    published_record_total: int | None = None
    difference_from_published_total: int | None = None


class SourceCode(BaseModel):
    files_sha256: dict[str, str] = Field(default_factory=dict)
    git_dirty: bool | None = None


class Manifest(BaseModel):
    status: Literal["completed"]
    started_at_utc: str
    completed_at_utc: str
    dataset: Dataset
    seed: int
    models: list[Model] = Field(min_length=1)
    feature_sets: dict[FeatureSet, list[str]] = Field(min_length=1)
    target_specificity: Probability
    fit_cohort: Literal["development"]
    threshold_selection_cohort: Literal["validation"]
    evaluated_cohorts: list[Literal["validation", "external", "test"]] = Field(min_length=1)
    evaluate_holdouts: bool
    cohort_counts: dict[Cohort, Counts]
    versions: dict[str, str]
    source_code: SourceCode
    preparation_provenance: Provenance | None = None
    evaluation_note: str

    @model_validator(mode="after")
    def complete_cohorts(self):
        required = {"development", "validation", *self.evaluated_cohorts}
        if not required.issubset(self.cohort_counts):
            raise ValueError("Missing cohort counts")
        if any(self.cohort_counts[name].n == 0 for name in required):
            raise ValueError("Required cohorts must contain records")
        if sum(count.n for count in self.cohort_counts.values()) != self.dataset.rows:
            raise ValueError("Dataset count mismatch")
        if len(set(self.models)) != len(self.models) or len(set(self.evaluated_cohorts)) != len(self.evaluated_cohorts):
            raise ValueError("Duplicate experiment definitions")
        return self


class Report(BaseModel):
    run_id: str
    manifest: Manifest
    results: list[Result]

    @model_validator(mode="after")
    def consistent_report(self):
        seen = set()
        if not self.manifest.evaluate_holdouts and any(r.cohort != "validation" for r in self.results):
            raise ValueError("Unapproved holdouts in report")
        for row in self.results:
            key = (row.feature_set, row.model, row.cohort, row.scope, row.site)
            if key in seen:
                raise ValueError("Duplicate aggregate result")
            seen.add(key)
            if row.model not in self.manifest.models or row.feature_set not in self.manifest.feature_sets:
                raise ValueError("Undeclared experiment")
            if row.cohort not in self.manifest.evaluated_cohorts:
                raise ValueError("Undeclared cohort")
            if row.scope == "overall":
                expected = self.manifest.cohort_counts[row.cohort]
                if row.site is not None or any(getattr(row.metrics, field) != getattr(expected, field) for field in ("n", "n_positive", "n_negative")):
                    raise ValueError("Overall count mismatch")
            elif not row.site:
                raise ValueError("Site result has no site")
            if sum(bin.n for bin in row.reliability_bins) != row.metrics.n:
                raise ValueError("Reliability counts do not add up")
        for feature_set in self.manifest.feature_sets:
            for model in self.manifest.models:
                for cohort in self.manifest.evaluated_cohorts:
                    if (feature_set, model, cohort, "overall", None) not in seen:
                        raise ValueError("Missing experiment result")
                    rows = [r for r in self.results if (r.feature_set, r.model, r.cohort) == (feature_set, model, cohort)]
                    overall = next(r for r in rows if r.scope == "overall")
                    sites = [r for r in rows if r.scope == "site"]
                    for field in ("n", "n_positive", "n_negative", "true_positives", "true_negatives", "false_positives", "false_negatives"):
                        if sum(getattr(r.metrics, field) for r in sites) != getattr(overall.metrics, field):
                            raise ValueError("Site totals disagree with overall results")
                    if any(r.metrics.threshold != overall.metrics.threshold for r in sites):
                        raise ValueError("Site threshold differs from pooled threshold")
                thresholds = {r.metrics.threshold for r in self.results if (r.feature_set, r.model) == (feature_set, model)}
                if len(thresholds) != 1:
                    raise ValueError("Threshold changed between cohorts")
        return self


def _load_report(run_id: str) -> Report:
    root = Path(RESEARCH_REPORTS_DIR).resolve()
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Research run not found")
    directory = (root / run_id).resolve()
    if directory.parent != root or not directory.is_dir():
        raise HTTPException(status_code=404, detail="Research run not found")
    paths = [directory / name for name in ("manifest.json", "metrics.json")]
    if any(path.resolve().parent != directory or not path.is_file() for path in paths):
        raise HTTPException(status_code=404, detail="Research run not found")
    try:
        manifest, metrics = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        # Explicit response models omit raw paths, records, model artifacts and unknown fields.
        return Report(run_id=run_id, manifest=manifest, results=metrics["results"])
    except (OSError, ValueError, KeyError, TypeError, ValidationError):
        raise HTTPException(status_code=503, detail="Research report is incomplete or invalid") from None


@router.get("")
def list_runs():
    root = Path(RESEARCH_REPORTS_DIR)
    runs, unavailable = [], 0
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not RUN_ID.fullmatch(directory.name):
                continue
            try:
                report = _load_report(directory.name)
            except HTTPException:
                unavailable += 1
                continue
            runs.append({"run_id": report.run_id, "completed_at_utc": report.manifest.completed_at_utc,
                         "records": report.manifest.dataset.rows,
                         "experiments": len(report.manifest.models) * len(report.manifest.feature_sets)})
    return {"runs": sorted(runs, key=lambda row: row["completed_at_utc"], reverse=True), "unavailable_runs": unavailable}


@router.get("/{run_id}", response_model=Report)
def get_run(run_id: str, download: bool = False):
    report = _load_report(run_id)
    if download:
        return JSONResponse(report.model_dump(mode="json"), headers={
            "Content-Disposition": f'attachment; filename="{run_id}-aggregate-results.json"',
        })
    return report


@router.get("/{run_id}/profile", response_model=DatasetProfile)
def get_profile(run_id: str, download: bool = False):
    report = _load_report(run_id)
    directory = (Path(RESEARCH_REPORTS_DIR) / run_id).resolve()
    path = directory / "profile.json"
    if not path.is_file() or path.resolve().parent != directory:
        raise HTTPException(status_code=404, detail="Dataset profile is not available")
    try:
        profile = DatasetProfile.model_validate_json(path.read_text(encoding="utf-8"))
        if profile.dataset_sha256 != report.manifest.dataset.sha256:
            raise ValueError("Dataset checksum mismatch")
        expected_cohorts = {"development", *report.manifest.evaluated_cohorts}
        if {g.cohort for g in profile.groups} != expected_cohorts:
            raise ValueError("Profile contains different cohorts")
        for group in profile.groups:
            if group.scope == "overall":
                expected = report.manifest.cohort_counts[group.cohort]
                if group.n != expected.n or group.n_positive != expected.n_positive:
                    raise ValueError("Profile counts differ from benchmark")
            elif group.cohort != "development":
                expected = next((r.metrics for r in report.results if r.cohort == group.cohort and r.scope == "site" and r.site == group.site), None)
                if expected is None or group.n != expected.n or group.n_positive != expected.n_positive:
                    raise ValueError("Profile site counts differ from benchmark")
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=503, detail="Dataset profile is incomplete or invalid") from None
    if download:
        return JSONResponse(profile.model_dump(mode="json"), headers={
            "Content-Disposition": f'attachment; filename="{run_id}-dataset-profile.json"',
        })
    return profile


@router.get("/{run_id}/cross-validation", response_model=CrossValidationReport)
def get_cross_validation(run_id: str, download: bool = False):
    report = _load_report(run_id)
    directory = (Path(RESEARCH_REPORTS_DIR) / run_id).resolve()
    path = directory / "cross_validation.json"
    if not path.is_file() or path.resolve().parent != directory:
        raise HTTPException(status_code=404, detail="Development checks are not available")
    try:
        cv = CrossValidationReport.model_validate_json(path.read_text(encoding="utf-8"))
        expected = report.manifest.cohort_counts["development"]
        if cv.dataset_sha256 != report.manifest.dataset.sha256 or cv.development_n != expected.n or cv.development_positive != expected.n_positive:
            raise ValueError("Development dataset differs from benchmark")
        if cv.models != report.manifest.models or cv.feature_sets != report.manifest.feature_sets:
            raise ValueError("Experiment definitions differ from benchmark")
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(status_code=503, detail="Development checks are incomplete or invalid") from None
    if download:
        return JSONResponse(cv.model_dump(mode="json"), headers={
            "Content-Disposition": f'attachment; filename="{run_id}-development-checks.json"',
        })
    return cv

"""Reproducible CBC screening baselines with explicit held-out cohort evaluation.

This module consumes the canonical CSV produced by ``research.leukoalert``.
It does not establish patient-disjoint evaluation or pre-diagnosis prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .schema import FEATURE_SETS


METADATA_COLUMNS = ("sample_id", "site", "cohort", "target")
COHORTS = ("development", "validation", "external", "test")
AVAILABLE_MODELS = ("logistic", "rf", "xgboost")
EVALUATION_NOTE = (
    "Research screening benchmark only. The public release has sample identifiers "
    "but no patient identifiers or diagnosis dates. Repeated patients and "
    "patient overlap across cohorts cannot be verified. Results do not establish "
    "early detection, clinical utility, or patient-disjoint generalization. "
    "Thresholds are selected on validation records; validation performance is "
    "therefore not an unbiased estimate for the selected operating point. "
    "External/test evaluation must be reserved until the model configuration "
    "and threshold-selection procedure are frozen."
)


def threshold_at_specificity(
    y_true: Sequence[int], probabilities: Sequence[float], target_specificity: float
) -> float:
    """Smallest threshold attaining specificity, with positive = score >= threshold.

    Moving immediately above the relevant negative score handles tied scores
    conservatively. The returned threshold can exceed one when all negatives
    have probability one; this intentionally predicts no positives.
    """
    if not 0 < target_specificity <= 1:
        raise ValueError("target_specificity must be greater than zero and at most one")
    y = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores) or not len(y):
        raise ValueError("Labels and probabilities must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Threshold labels must be binary (0/1)")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and between zero and one")
    negatives = np.sort(scores[y == 0])
    if not len(negatives):
        raise ValueError("Validation requires negative examples to select specificity")
    rank = math.ceil(target_specificity * len(negatives)) - 1
    return float(np.nextafter(negatives[rank], np.inf))


def classification_metrics(
    y_true: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict[str, Any]:
    """Aggregate metrics; undefined denominators/ranking metrics are JSON null."""
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positive = probabilities >= threshold
    tp = int(np.sum((y == 1) & positive))
    tn = int(np.sum((y == 0) & ~positive))
    fp = int(np.sum((y == 0) & positive))
    fn = int(np.sum((y == 1) & ~positive))
    n_positive, n_negative = tp + fn, tn + fp
    two_classes = bool(n_positive and n_negative)
    return {
        "n": int(len(y)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, probabilities)) if two_classes else None,
        "average_precision": (
            float(average_precision_score(y, probabilities)) if two_classes else None
        ),
        "brier": float(brier_score_loss(y, probabilities)),
        "sensitivity": tp / n_positive if n_positive else None,
        "specificity": tn / n_negative if n_negative else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_positives_per_1000_negatives": 1000 * fp / n_negative if n_negative else None,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "threshold": threshold,
    }


def reliability_bins(
    y_true: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Equal-width reliability bins including empty bins and the probability one."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    assignments = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    bins = []
    for index in range(n_bins):
        included = assignments == index
        count = int(included.sum())
        bins.append(
            {
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "upper_inclusive": index == n_bins - 1,
                "n": count,
                "mean_predicted_probability": float(scores[included].mean()) if count else None,
                "observed_positive_fraction": float(y[included].mean()) if count else None,
            }
        )
    return bins


def build_pipeline(model_name: str, seed: int = 42) -> Pipeline:
    """Construct unfitted estimators; optional XGBoost is imported only on request."""
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "prevalence":
        estimator = DummyClassifier(strategy="prior")
    elif model_name == "logistic":
        steps.append(("scaler", StandardScaler()))
        estimator = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=-1
        )
    elif model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError("XGBoost was requested but is not installed; install xgboost first") from exc
        estimator = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=1.0,
            colsample_bytree=1.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def _load_dataset(path: Path, feature_set_names: Sequence[str]) -> pd.DataFrame:
    # Read IDs as strings so they cannot become numeric model features accidentally.
    frame = pd.read_csv(path, dtype={"sample_id": "string", "site": "string", "cohort": "string"})
    features = list(dict.fromkeys(column for name in feature_set_names for column in FEATURE_SETS[name]))
    if not features or any(not FEATURE_SETS[name] for name in feature_set_names):
        raise ValueError("Every requested feature set must contain at least one feature")
    forbidden = set(features).intersection(METADATA_COLUMNS)
    if forbidden:
        raise ValueError(f"Feature schema includes forbidden metadata: {sorted(forbidden)}")
    required = set(METADATA_COLUMNS).union(features)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    for column in ("sample_id", "site", "cohort"):
        if frame[column].isna().any() or frame[column].str.strip().eq("").any():
            raise ValueError(f"{column} values must be present and nonempty")
    invalid = set(frame["cohort"].unique()).difference(COHORTS)
    if invalid:
        raise ValueError(f"Unknown cohort values: {sorted(invalid)}")
    if frame["target"].isna().any() or not frame["target"].isin([0, 1]).all():
        raise ValueError("target must contain only binary 0/1 labels without missing values")
    frame["target"] = frame["target"].astype(int)
    for feature in features:
        try:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Feature {feature} contains nonnumeric values") from exc
        if np.isinf(frame[feature].to_numpy(dtype=float)).any():
            raise ValueError(f"Feature {feature} contains infinite values")
    for cohort in ("development", "validation"):
        subset = frame.loc[frame["cohort"] == cohort]
        if subset.empty or subset["target"].nunique() != 2:
            raise ValueError(f"{cohort} must contain both target classes")
    development = frame.loc[frame["cohort"] == "development", features]
    unusable = development.columns[development.isna().all()].tolist()
    if unusable:
        raise ValueError(f"Features entirely missing from development data: {unusable}")
    return frame


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_identity() -> dict[str, Any]:
    source_dir = Path(__file__).resolve().parent
    hashes = {
        name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        for name in ("benchmark.py", "schema.py", "leukoalert.py")
        if (source_dir / name).is_file()
    }
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_dir.parents[1],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"files_sha256": hashes, "git_dirty": dirty}


def _preparation_provenance(dataset: Path, dataset_digest: str) -> dict[str, Any] | None:
    path = dataset.parent / "manifest.json"
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError("Cannot read preparation manifest alongside dataset") from exc
    if not isinstance(metadata, dict) or metadata.get("prepared_csv_sha256") != dataset_digest:
        raise ValueError("Preparation manifest checksum does not match the dataset")
    # Preserve source attribution/schema, never copy arbitrary per-record metadata.
    allowed = (
        "schema_version", "source_url", "source_doi", "source_license", "source_archive_sha256",
        "prepared_csv_sha256", "verified_official_archive", "records", "published_record_total",
        "difference_from_published_total", "feature_sets", "column_mapping", "limitations", "processing",
    )
    provenance = {key: metadata[key] for key in allowed if key in metadata}
    provenance["preparation_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return provenance


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "xgboost"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def _leaderboard(results: list[dict[str, Any]], evaluate_holdouts: bool) -> str:
    def display(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    lines = [
        "# CBC screening research benchmark", "", EVALUATION_NOTE, "",
        "Models use fixed hyperparameters and fit development records only. Median imputation "
        "and logistic-regression scaling are learned only from development data. "
        "No calibration correction or hyperparameter tuning is performed.", "",
        "External/test cohorts were evaluated with the explicit holdout flag."
        if evaluate_holdouts else "External/test cohorts were not evaluated.", "",
        "Rows retain the prespecified experiment order; this table does not select a winner.", "",
        "| Feature set | Model | Cohort | N | Prevalence | AUROC | AP | Brier | Sensitivity | Specificity | Precision | FP / 1,000 negatives |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        if row["scope"] != "overall":
            continue
        m = row["metrics"]
        lines.append(
            f"| {row['feature_set']} | {row['model']} | {row['cohort']} | {m['n']} | "
            + " | ".join(display(m[name]) for name in (
                "prevalence", "auroc", "average_precision", "brier", "sensitivity",
                "specificity", "precision", "false_positives_per_1000_negatives",
            )) + " |"
        )
    lines.extend([
        "", "AP = average precision. Undefined metrics are shown as — and saved as null. "
        "Specificity is targeted on validation overall and may differ by site or cohort.", "",
        "## Performance by site", "",
        "| Feature set | Model | Cohort | Site | N | Sensitivity | Specificity | FP / 1,000 negatives |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for row in results:
        if row["scope"] != "site":
            continue
        m = row["metrics"]
        safe_site = str(row["site"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['feature_set']} | {row['model']} | {row['cohort']} | {safe_site} | {m['n']} | "
            + " | ".join(display(m[name]) for name in (
                "sensitivity", "specificity", "false_positives_per_1000_negatives",
            )) + " |"
        )
    lines.extend(["", "Full ranking metrics, confusion counts, and ten-bin reliability data "
                  "for every site are in `metrics.json`. No individual predictions are exported.", ""])
    return "\n".join(lines)


def run_benchmark(
    dataset: str | Path,
    output: str | Path,
    *,
    models: Sequence[str] = ("logistic", "rf"),
    feature_sets: Sequence[str] | None = None,
    target_specificity: float = 0.95,
    seed: int = 42,
    evaluate_holdouts: bool = False,
) -> dict[str, Any]:
    """Fit baseline pipelines and save aggregate, reproducible experiment artifacts."""
    dataset, output = Path(dataset), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory; previous runs are never overwritten")
    models = list(models)
    feature_sets = list(FEATURE_SETS) if feature_sets is None else list(feature_sets)
    if not models or len(set(models)) != len(models) or set(models).difference(AVAILABLE_MODELS):
        raise ValueError(f"Specify distinct supported models: {', '.join(AVAILABLE_MODELS)}")
    if not feature_sets or len(set(feature_sets)) != len(feature_sets) or set(feature_sets).difference(FEATURE_SETS):
        raise ValueError(f"Specify distinct feature sets: {', '.join(FEATURE_SETS)}")
    if not 0 < target_specificity <= 1:
        raise ValueError("target_specificity must be greater than zero and at most one")
    dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    frame = _load_dataset(dataset, feature_sets)
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != dataset_digest:
        raise ValueError("Dataset changed while being read; retry with a stable file")
    provenance = _preparation_provenance(dataset, dataset_digest)
    if evaluate_holdouts:
        absent = [cohort for cohort in ("external", "test") if not frame["cohort"].eq(cohort).any()]
        if absent:
            raise ValueError(f"Requested holdout evaluation, but cohorts are absent: {absent}")
    # Check optional dependencies before reserving the output directory.
    for model in models:
        build_pipeline(model, seed)
    output.mkdir(parents=True, exist_ok=True)
    # Atomic exclusive creation also prevents two runs from sharing an output directory.
    lock = output / ".run.lock"
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(datetime.now(timezone.utc).isoformat())
    except FileExistsError as exc:
        raise ValueError("Output directory is already in use by another run") from exc
    (output / "models").mkdir()
    evaluation_cohorts = ["validation"] + (["external", "test"] if evaluate_holdouts else [])
    manifest: dict[str, Any] = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"filename": dataset.name, "sha256": dataset_digest, "rows": len(frame)},
        "git_revision": _git_revision(), "versions": _versions(), "seed": seed,
        "source_code": _source_identity(), "preparation_provenance": provenance,
        "feature_sets": {name: list(FEATURE_SETS[name]) for name in feature_sets},
        "excluded_metadata_columns": list(METADATA_COLUMNS),
        "models": ["prevalence"] + models, "target_specificity": target_specificity,
        "fit_cohort": "development", "threshold_selection_cohort": "validation",
        "evaluated_cohorts": evaluation_cohorts, "evaluate_holdouts": evaluate_holdouts,
        "evaluation_note": EVALUATION_NOTE,
        "cohort_counts": {
            cohort: {"n": int(len(subset)), "n_positive": int(subset.target.sum()),
                     "n_negative": int(len(subset) - subset.target.sum())}
            for cohort, subset in frame.groupby("cohort", sort=True)
        },
        "experiments": [],
    }
    _write_json(output / "manifest.json", manifest)
    train = frame.loc[frame.cohort == "development"]
    validation = frame.loc[frame.cohort == "validation"]
    results: list[dict[str, Any]] = []
    try:
        for feature_set in feature_sets:
            columns = list(FEATURE_SETS[feature_set])
            for model_name in ["prevalence"] + models:
                pipeline = build_pipeline(model_name, seed)
                pipeline.fit(train[columns], train.target)
                validation_probabilities = pipeline.predict_proba(validation[columns])[:, 1]
                threshold = threshold_at_specificity(validation.target, validation_probabilities, target_specificity)
                bundle_name = f"{feature_set}__{model_name}.joblib"
                joblib.dump({
                    "pipeline": pipeline, "feature_columns": columns,
                    "feature_set": feature_set, "model": model_name, "threshold": threshold,
                    "threshold_selection_cohort": "validation", "target_specificity": target_specificity,
                    "dataset_sha256": dataset_digest, "evaluation_note": EVALUATION_NOTE,
                }, output / "models" / bundle_name)
                manifest["experiments"].append({
                    "feature_set": feature_set, "model": model_name, "threshold": threshold,
                    "model_bundle": f"models/{bundle_name}",
                    "parameters": {name: estimator.get_params(deep=False) for name, estimator in pipeline.steps},
                    "training_missing_counts": {name: int(train[name].isna().sum()) for name in columns},
                })
                for cohort in evaluation_cohorts:
                    subset = frame.loc[frame.cohort == cohort]
                    probabilities = (validation_probabilities if cohort == "validation"
                                     else pipeline.predict_proba(subset[columns])[:, 1])
                    groups = [("overall", None, np.ones(len(subset), dtype=bool))]
                    groups += [("site", str(site), subset.site.eq(site).to_numpy(dtype=bool))
                               for site in sorted(subset.site.unique())]
                    for scope, site, selected in groups:
                        y = subset.target.to_numpy()[selected]
                        scores = probabilities[selected]
                        results.append({
                            "feature_set": feature_set, "model": model_name, "cohort": cohort,
                            "scope": scope, "site": site,
                            "metrics": classification_metrics(y, scores, threshold),
                            "reliability_bins": reliability_bins(y, scores),
                        })
        _write_json(output / "metrics.json", {"evaluation_note": EVALUATION_NOTE, "results": results})
        (output / "leaderboard.md").write_text(_leaderboard(results, evaluate_holdouts), encoding="utf-8")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "manifest.json", manifest)
    except Exception:
        manifest["status"] = "failed"
        # Avoid recording exception text: parsers may include individual source values.
        manifest["failure_note"] = "Run failed; retain this directory and use a new output directory when retrying."
        _write_json(output / "manifest.json", manifest)
        raise
    return {"manifest": manifest, "results": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Canonical prepared CBC CSV")
    parser.add_argument("--output", type=Path, required=True, help="New or empty experiment directory")
    parser.add_argument("--models", nargs="+", choices=AVAILABLE_MODELS, default=["logistic", "rf"])
    parser.add_argument("--feature-sets", nargs="+", choices=list(FEATURE_SETS), default=list(FEATURE_SETS))
    parser.add_argument("--target-specificity", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-holdouts", action="store_true", help="Explicitly evaluate external/test cohorts after freezing the experiment")
    arguments = parser.parse_args(argv)
    try:
        result = run_benchmark(
            arguments.dataset, arguments.output, models=arguments.models,
            feature_sets=arguments.feature_sets, target_specificity=arguments.target_specificity,
            seed=arguments.seed, evaluate_holdouts=arguments.evaluate_holdouts,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved {len(result['manifest']['experiments'])} experiments to {arguments.output.resolve()}")
    print("Evaluated cohorts: " + ", ".join(result["manifest"]["evaluated_cohorts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

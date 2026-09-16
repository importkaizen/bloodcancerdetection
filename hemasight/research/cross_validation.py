"""Development-only stability and shuffled-training-label diagnostics.

This is a descriptive record-level check, not a patient-disjoint estimate or
a formal permutation test. No model is selected, tuned, saved, or deployed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .benchmark import _load_dataset, _versions, build_pipeline
from .cv_schema import CrossValidationReport
from .schema import FEATURE_SETS


def run_cross_validation(dataset: Path, run_directory: Path, *, n_splits: int = 5, seed: int = 42):
    dataset, run_directory = Path(dataset), Path(run_directory)
    output = run_directory / "cross_validation.json"
    if output.exists():
        raise FileExistsError("Cross-validation report already exists; preserve the saved run")
    if not 2 <= n_splits <= 10 or not 0 <= seed <= 2**32 - 100:
        raise ValueError("Use 2–10 folds and a nonnegative 32-bit seed")
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("A completed benchmark is required")
    models, feature_sets = manifest["models"], manifest["feature_sets"]
    if not models or len(set(models)) != len(models) or "logistic" not in models:
        raise ValueError("Distinct models including logistic are required")
    if not feature_sets or any(name not in FEATURE_SETS or columns != FEATURE_SETS[name] for name, columns in feature_sets.items()):
        raise ValueError("Only the fixed canonical feature sets are supported")
    # Check availability before doing any training.
    for name in models:
        build_pipeline(name, seed)
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if digest != manifest["dataset"]["sha256"]:
        raise ValueError("Dataset checksum differs from completed benchmark")
    frame = _load_dataset(dataset, list(feature_sets))
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != digest:
        raise ValueError("Dataset changed while reading")
    development = frame.loc[frame.cohort == "development"].copy()
    del frame  # Validation/external/test records never reach a model or splitter.
    target = development.target.to_numpy()
    expected = manifest["cohort_counts"]["development"]
    if len(target) != expected["n"] or int(target.sum()) != expected["n_positive"]:
        raise ValueError("Development counts differ from completed benchmark")
    if min(np.bincount(target, minlength=2)) < n_splits:
        raise ValueError("Each development class must have at least n_splits records")
    folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(development, target))
    started = datetime.now(timezone.utc).isoformat()
    results = []
    for feature_set, columns in feature_sets.items():
        experiments = [(name, "original") for name in models] + [("logistic", "shuffled_training_labels")]
        for model, control in experiments:
            for fold, (train_index, evaluation_index) in enumerate(folds, start=1):
                train = development.iloc[train_index]
                evaluation = development.iloc[evaluation_index]
                # Fit imputation and scaling independently on each training partition.
                pipeline = build_pipeline(model, seed + fold)
                train_labels = target[train_index].copy()
                if control == "shuffled_training_labels":
                    train_labels = np.random.default_rng(seed + fold).permutation(train_labels)
                pipeline.fit(train.loc[:, columns], train_labels)
                probabilities = pipeline.predict_proba(evaluation.loc[:, columns])[:, 1]
                labels = target[evaluation_index]
                results.append({"feature_set": feature_set, "model": model, "control": control,
                                "fold": fold, "train_n": len(train), "train_positive": int(train_labels.sum()),
                                "evaluation_n": len(evaluation), "evaluation_positive": int(labels.sum()),
                                "auroc": float(roc_auc_score(labels, probabilities)),
                                "average_precision": float(average_precision_score(labels, probabilities)),
                                "brier": float(brier_score_loss(labels, probabilities))})
            print(f"Completed {feature_set}: {model} ({control}), {n_splits} folds", flush=True)
    source = Path(__file__).resolve().parent
    report = CrossValidationReport(
        schema_version="development_cv_v1", status="completed", dataset_sha256=digest,
        development_n=len(target), development_positive=int(target.sum()),
        started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                       for name in ("cross_validation.py", "cv_schema.py", "benchmark.py", "schema.py")},
        versions=_versions(), seed=seed, n_splits=n_splits,
        split_strategy="stratified_kfold_shuffled_records", fit_cohort="development", evaluation_cohort="development",
        shuffled_control="logistic_training_labels_per_fold", models=models, feature_sets=feature_sets, results=results,
    )
    with output.open("x", encoding="utf-8") as stream:
        stream.write(report.model_dump_json(indent=2) + "\n")
    return report.model_dump(mode="json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_cross_validation(args.dataset, args.run_directory, n_splits=args.folds, seed=args.seed)


if __name__ == "__main__":
    main()

"""Train research-only CBC classifiers with train-fitted preprocessing."""
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hemasight.config import ML_MODELS_DIR, MODEL_CONFIG_PATH, RISK_MODEL_PATH, SCALER_PATH

FEATURE_COLUMNS = [
    "wbc",
    "rbc",
    "platelets",
    "hemoglobin",
    "lymphocytes",
    "wbc_trend",
    "platelet_var",
    "hemoglobin_drop_rate",
    "lymphocyte_spike",
    "wbc_rolling_avg",
    "rbc_rolling_avg",
    "platelets_rolling_avg",
    "hemoglobin_rolling_avg",
]
DEFAULT_LABEL_COL = "label"
MODEL_VERSION = "v2"
THRESHOLDS = {"LOW": 0.33, "MEDIUM": 0.66}


def _read_table(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def _validate_training_data(df: pd.DataFrame, label_col: str) -> tuple:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in data.")
    if label_col in FEATURE_COLUMNS:
        raise ValueError("The label column must not also be a model feature.")
    if df.empty:
        raise ValueError("Training data is empty.")
    try:
        X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
        y = pd.to_numeric(df[label_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Features and labels must be numeric; missing feature values are allowed.") from exc
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Features must not contain infinite values.")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must be non-missing binary values 0 and 1.")
    if y.nunique() != 2:
        raise ValueError("Training data must contain both label classes 0 and 1.")
    # Preserve NaNs: fitting the imputer before the split leaks holdout information.
    return X, y.astype(int)


def load_training_data(path: str, label_col: str = DEFAULT_LABEL_COL) -> tuple:
    """Load ordered numeric features and binary labels without fitting preprocessing."""
    return _validate_training_data(_read_table(path), label_col)


def _split_indices(y, test_size, random_state, groups=None) -> tuple:
    indices = np.arange(len(y))
    if groups is None:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        details = {
            "method": "stratified_row_split",
            "note": "Exploratory row-level evaluation only; patient overlap cannot be excluded without patient identifiers.",
        }
    else:
        if groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError("The patient group column must not contain missing identifiers.")
        if groups.nunique() < 2:
            raise ValueError("At least two patient groups are required.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, y, groups))
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        details = {
            "method": "patient_group_split",
            "train_group_count": len(train_groups),
            "test_group_count": len(test_groups),
            "groups_disjoint": train_groups.isdisjoint(test_groups),
            "note": "Patient-disjoint internal holdout; this does not establish external hospital generalization.",
        }
    if y.iloc[train_idx].nunique() != 2:
        raise ValueError("The training split needs both classes; revise the cohort or split settings.")
    return train_idx, test_idx, details


def train(
    data_path: str,
    label_col: str = DEFAULT_LABEL_COL,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
    *,
    patient_group_col: str | None = None,
    feature_version: str = "v1",
) -> dict:
    if model_type not in {"xgboost", "rf"}:
        raise ValueError("model_type must be 'xgboost' or 'rf'.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if feature_version not in {"v1", "v2"}:
        raise ValueError("feature_version must be 'v1' or 'v2'.")
    df = _read_table(data_path)
    X, y = _validate_training_data(df, label_col)
    if "feature_version" in df.columns:
        if df["feature_version"].isna().any() or set(df["feature_version"].unique()) != {feature_version}:
            raise ValueError("Training rows must all match the declared feature_version.")
    groups = None
    if patient_group_col is not None:
        if patient_group_col not in df.columns:
            raise ValueError(f"Patient group column '{patient_group_col}' not in data.")
        if patient_group_col in FEATURE_COLUMNS or patient_group_col == label_col:
            raise ValueError("The patient group column must be separate from features and labels.")
        groups = df[patient_group_col]
    train_idx, test_idx, split_details = _split_indices(y, test_size, random_state, groups)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    empty_features = X_train.columns[X_train.isna().all()].tolist()
    if empty_features:
        raise ValueError(f"Features entirely missing in the training split: {', '.join(empty_features)}")
    preprocessing = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_scaled = preprocessing.fit_transform(X_train)
    X_test_scaled = preprocessing.transform(X_test)
    packages = {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib"]}
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=100, max_depth=5, random_state=random_state, eval_metric="logloss", n_jobs=1)
        packages["xgboost"] = version("xgboost")
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=1)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    probabilities = clf.predict_proba(X_test_scaled)[:, list(clf.classes_).index(1)]
    both_test_classes = y_test.nunique() == 2
    metrics = {
        "auroc": float(roc_auc_score(y_test, probabilities)) if both_test_classes else None,
        "average_precision": float(average_precision_score(y_test, probabilities)) if both_test_classes else None,
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "test_positive_fraction": float(y_test.mean()),
        "classification_report": report,
        "note": None if both_test_classes else "AUROC and average precision are not reported for a single-class holdout.",
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "random_state": random_state,
        "test_size": test_size,
        "label_column": label_col,
        "patient_group_column": patient_group_col,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_positive_fraction": float(y_train.mean()),
        "split": split_details,
        "model_params": clf.get_params(),
        "package_versions": packages,
    }
    config = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_schema": [{"name": c, "dtype": "float64", "nullable": True} for c in FEATURE_COLUMNS],
        "feature_version": feature_version,
        "model_version": MODEL_VERSION,
        "thresholds": THRESHOLDS,
        "threshold_note": "Legacy demonstration cutoffs; not clinically validated operating thresholds.",
        "model_type": model_type,
        "artifact_format_version": 2,
        "preprocessing": "train_fitted_median_imputer_and_standard_scaler",
        "intended_use": "research_only",
        "run": metadata,
        "evaluation": metrics,
    }
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, RISK_MODEL_PATH)
    # Preserve the artifact filename while saving the complete preprocessing pipeline.
    joblib.dump(preprocessing, SCALER_PATH)
    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, allow_nan=False)
    return {"classification_report": report, "model_version": MODEL_VERSION, "metrics": metrics, "run": metadata}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to CSV or Parquet")
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--model-type", choices=["xgboost", "rf"], default="xgboost")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--patient-group-col", help="Actual patient identifier column for a patient-disjoint holdout")
    parser.add_argument("--feature-version", choices=["v1", "v2"], default="v1", help="Version of precomputed longitudinal features; use v2 for newly derived features")
    args = parser.parse_args()
    print(json.dumps(train(
        args.data_path, label_col=args.label_col, model_type=args.model_type,
        test_size=args.test_size, random_state=args.random_state,
        patient_group_col=args.patient_group_col, feature_version=args.feature_version,
    ), indent=2, allow_nan=False))

"""Risk score inference: load model, compute score and level."""
import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from hemasight.config import MODEL_CONFIG_PATH, RISK_MODEL_PATH, SCALER_PATH

THRESHOLDS = {"LOW": 0.33, "MEDIUM": 0.66}
DEFAULT_MESSAGE = "Your blood patterns show unusual drift that may indicate early hematologic abnormalities."


def _load_artifacts():
    model = joblib.load(RISK_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(MODEL_CONFIG_PATH) as f:
        config = json.load(f)
    return model, scaler, config


def feature_row_to_vector(feature_row: dict, feature_columns: Optional[list] = None) -> np.ndarray:
    cols = [
        "wbc", "rbc", "platelets", "hemoglobin", "lymphocytes",
        "wbc_trend", "platelet_var", "hemoglobin_drop_rate", "lymphocyte_spike",
        "wbc_rolling_avg", "rbc_rolling_avg", "platelets_rolling_avg", "hemoglobin_rolling_avg",
    ]
    if feature_columns is not None:
        cols = feature_columns
    elif "feature_columns" in feature_row:
        cols = feature_row["feature_columns"]
    vec = []
    for c in cols:
        v = feature_row.get(c)
        if v is None or pd.isna(v):
            v = np.nan
        try:
            numeric = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{c}' must be numeric or missing.") from exc
        if np.isinf(numeric):
            raise ValueError(f"Feature '{c}' must not be infinite.")
        vec.append(numeric)
    return np.array(vec).reshape(1, -1)


def score_to_level(score: float, thresholds: Optional[dict] = None) -> str:
    if thresholds is None:
        thresholds = THRESHOLDS
    if score < thresholds.get("LOW", 0.33):
        return "LOW"
    if score < thresholds.get("MEDIUM", 0.66):
        return "MEDIUM"
    return "ELEVATED"


def get_message(level: str) -> str:
    if level == "LOW":
        return "Blood patterns are within expected range."
    if level == "MEDIUM":
        return "Some blood pattern drift observed; consider follow-up."
    return DEFAULT_MESSAGE


def compute_risk(feature_vector: np.ndarray, model=None, scaler=None, config=None) -> dict:
    if model is None or scaler is None or config is None:
        model, scaler, config = _load_artifacts()
    cols = config.get("feature_columns", [])
    X = np.asarray(feature_vector, dtype=float)
    if X.ndim != 2 or X.shape[0] != 1:
        raise ValueError("Feature vector must have shape (1, number_of_features).")
    if X.shape[1] != len(cols):
        raise ValueError(f"Feature vector length {X.shape[1]} != config columns {len(cols)}")
    if np.isinf(X).any():
        raise ValueError("Feature vector must not contain infinite values.")
    has_imputer = isinstance(scaler, Pipeline) and any(
        isinstance(step, SimpleImputer) for _, step in scaler.steps
    )
    if np.isnan(X).any() and not has_imputer:
        raise ValueError("This legacy artifact has no trained imputer; retrain before scoring missing features.")
    # New pipelines were fit with named columns. Preserve names and their saved order.
    transform_input = pd.DataFrame(X, columns=cols) if hasattr(scaler, "feature_names_in_") else X
    X_scaled = scaler.transform(transform_input)
    if not np.isfinite(X_scaled).all():
        raise ValueError("Preprocessing left non-finite features; retrain a complete preprocessing artifact.")
    if hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", [0, 1]))
        if 1 not in classes:
            raise ValueError("Model does not expose the expected positive label 1.")
        score = float(model.predict_proba(X_scaled)[0, classes.index(1)])
    else:
        score = float(model.predict(X_scaled)[0])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model score must be a finite probability between 0 and 1.")
    thresholds = config.get("thresholds", THRESHOLDS)
    level = score_to_level(score, thresholds)
    return {
        "score": round(score, 4),
        "level": level,
        "model_version": config.get("model_version", "v1"),
        "message": get_message(level),
    }


def compute_risk_for_feature_id(feature_id: int) -> Optional[dict]:
    from hemasight.db.models import Feature, get_engine
    from sqlalchemy.orm import sessionmaker
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        row = db.query(Feature).filter(Feature.id == feature_id).first()
        if not row:
            return None
        row_dict = {c.key: getattr(row, c.key) for c in row.__table__.columns}
        model, scaler, config = _load_artifacts()
        expected_version = config.get("feature_version", "v1")
        if row_dict.get("feature_version") != expected_version:
            raise ValueError(
                f"Feature version '{row_dict.get('feature_version')}' does not match model feature version "
                f"'{expected_version}'; retrain on matching derived features."
            )
        vec = feature_row_to_vector(row_dict, feature_columns=config["feature_columns"])
        return compute_risk(vec, model=model, scaler=scaler, config=config)
    finally:
        db.close()

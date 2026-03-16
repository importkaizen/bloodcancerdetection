"""Risk score inference: load model, compute score and level."""
import json
from typing import Optional

import joblib
import numpy as np

from hemasight.config import MODEL_CONFIG_PATH, RISK_MODEL_PATH, SCALER_PATH

THRESHOLDS = {"LOW": 0.33, "MEDIUM": 0.66}
DEFAULT_MESSAGE = "Your blood patterns show unusual drift that may indicate early hematologic abnormalities."


def _load_artifacts():
    model = joblib.load(RISK_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(MODEL_CONFIG_PATH) as f:
        config = json.load(f)
    return model, scaler, config


def feature_row_to_vector(feature_row: dict) -> np.ndarray:
    cols = [
        "wbc", "rbc", "platelets", "hemoglobin", "lymphocytes",
        "wbc_trend", "platelet_var", "hemoglobin_drop_rate", "lymphocyte_spike",
        "wbc_rolling_avg", "rbc_rolling_avg", "platelets_rolling_avg", "hemoglobin_rolling_avg",
    ]
    if "feature_columns" in feature_row:
        cols = feature_row["feature_columns"]
    vec = []
    for c in cols:
        v = feature_row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        vec.append(float(v))
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
    X = feature_vector
    if X.shape[1] != len(cols):
        raise ValueError(f"Feature vector length {X.shape[1]} != config columns {len(cols)}")
    X_scaled = scaler.transform(X)
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X_scaled)[0, 1])
    else:
        score = float(model.predict(X_scaled)[0])
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
        vec = feature_row_to_vector(row_dict)
        return compute_risk(vec)
    finally:
        db.close()

"""Feature extraction Celery tasks."""
from typing import List, Optional

import numpy as np
from celery import Celery
from sqlalchemy import asc
from sqlalchemy.orm import sessionmaker

from hemasight.config import CELERY_BROKER_URL
from hemasight.db.models import BloodTest, Feature, get_engine

app = Celery(
    "hemasight",
    broker=CELERY_BROKER_URL,
)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.result_backend = None

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
FEATURE_VERSION = "v1"
WINDOW_SIZE = 5  # last N tests for rolling/trend


def _trend_slope(values: List[float]) -> Optional[float]:
    """Linear regression slope over values (time index 0..n-1)."""
    if not values or len(values) < 2:
        return None
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    x = np.arange(len(clean), dtype=float)
    y = np.array(clean)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _variance(values: List[float]) -> Optional[float]:
    if not values or len(values) < 2:
        return None
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    return float(np.var(clean))


def _rolling_avg(values: List[float], window: int = 3) -> Optional[float]:
    if not values:
        return None
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    take = clean[-window:] if len(clean) >= window else clean
    return float(np.mean(take))


def compute_features_for_blood_test(db, blood_test: BloodTest, history: List[BloodTest]) -> Feature:
    """Build feature row from this blood test and patient history."""
    patient_id = blood_test.patient_id
    wbc_vals = [bt.wbc for bt in history]
    rbc_vals = [bt.rbc for bt in history]
    platelet_vals = [bt.platelets for bt in history]
    hemoglobin_vals = [bt.hemoglobin for bt in history]
    lymphocyte_vals = [bt.lymphocytes for bt in history]

    feat = Feature(
        patient_id=patient_id,
        blood_test_id=blood_test.id,
        feature_version=FEATURE_VERSION,
        wbc_trend=_trend_slope(wbc_vals),
        platelet_var=_variance(platelet_vals),
        hemoglobin_drop_rate=-_trend_slope(hemoglobin_vals) if _trend_slope(hemoglobin_vals) is not None else None,
        lymphocyte_spike=_trend_slope(lymphocyte_vals),
        wbc_rolling_avg=_rolling_avg(wbc_vals, WINDOW_SIZE),
        rbc_rolling_avg=_rolling_avg(rbc_vals, WINDOW_SIZE),
        platelets_rolling_avg=_rolling_avg(platelet_vals, WINDOW_SIZE),
        hemoglobin_rolling_avg=_rolling_avg(hemoglobin_vals, WINDOW_SIZE),
        wbc=blood_test.wbc,
        rbc=blood_test.rbc,
        platelets=blood_test.platelets,
        hemoglobin=blood_test.hemoglobin,
        lymphocytes=blood_test.lymphocytes,
    )
    return feat


@app.task(bind=True)
def process_blood_test(self, blood_test_id: int) -> dict:
    """Load blood test, compute features, write to features table."""
    db = SessionLocal()
    try:
        blood_test = db.query(BloodTest).filter(BloodTest.id == blood_test_id).first()
        if not blood_test:
            return {"blood_test_id": blood_test_id, "status": "not_found"}
        patient_id = blood_test.patient_id
        history = (
            db.query(BloodTest)
            .filter(BloodTest.patient_id == patient_id, BloodTest.date <= blood_test.date)
            .order_by(asc(BloodTest.date))
            .limit(WINDOW_SIZE + 5)
            .all()
        )
        feat = compute_features_for_blood_test(db, blood_test, history)
        db.add(feat)
        db.commit()
        db.refresh(feat)
        compute_risk_score.delay(feat.id)
        compute_anomaly_score.delay(feat.id)
        return {"blood_test_id": blood_test_id, "feature_id": feat.id, "status": "ok"}
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


@app.task(bind=True)
def compute_risk_score(self, feature_id: int) -> dict:
    """Compute risk from feature row and write to risk_scores table. Skips if model not found."""
    from hemasight.config import RISK_MODEL_PATH
    from hemasight.db.models import RiskScore
    from hemasight.ml.inference import compute_risk_for_feature_id
    if not RISK_MODEL_PATH.exists():
        return {"feature_id": feature_id, "status": "skipped", "reason": "model_not_trained"}
    db = SessionLocal()
    try:
        result = compute_risk_for_feature_id(feature_id)
        if result is None:
            return {"feature_id": feature_id, "status": "not_found"}
        feat = db.query(Feature).filter(Feature.id == feature_id).first()
        if not feat:
            return {"feature_id": feature_id, "status": "not_found"}
        risk_row = RiskScore(
            patient_id=feat.patient_id,
            feature_id=feat.id,
            blood_test_id=feat.blood_test_id,
            score=result["score"],
            level=result["level"],
            model_version=result["model_version"],
            message=result["message"],
        )
        db.add(risk_row)
        db.commit()
        db.refresh(risk_row)
        return {"feature_id": feature_id, "risk_score_id": risk_row.id, "status": "ok"}
    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()


@app.task(bind=True)
def compute_anomaly_score(self, feature_id: int) -> dict:
    """Compute anomaly score from feature row and write to anomaly_scores table. Skips if model not found."""
    from hemasight.config import ML_MODELS_DIR
    from hemasight.db.models import AnomalyScore
    from hemasight.ml.anomaly import ANOMALY_MODEL_PATH, compute_anomaly_for_feature_id
    if not ANOMALY_MODEL_PATH.exists():
        return {"feature_id": feature_id, "status": "skipped", "reason": "anomaly_model_not_trained"}
    db = SessionLocal()
    try:
        result = compute_anomaly_for_feature_id(feature_id)
        if result is None:
            return {"feature_id": feature_id, "status": "not_found"}
        score, is_anomaly = result
        feat = db.query(Feature).filter(Feature.id == feature_id).first()
        if not feat:
            return {"feature_id": feature_id, "status": "not_found"}
        anomaly_row = AnomalyScore(
            patient_id=feat.patient_id,
            feature_id=feat.id,
            blood_test_id=feat.blood_test_id,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            model_version="isolation_forest_v1",
        )
        db.add(anomaly_row)
        db.commit()
        db.refresh(anomaly_row)
        return {"feature_id": feature_id, "anomaly_score_id": anomaly_row.id, "status": "ok"}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

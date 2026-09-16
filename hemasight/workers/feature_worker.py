"""Feature extraction Celery tasks."""
from functools import lru_cache
from typing import List

from celery import Celery
from kombu.exceptions import OperationalError
from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import sessionmaker

from hemasight.config import CELERY_BROKER_URL
from hemasight.db.models import BloodTest, Feature, get_engine
from hemasight.ml.features import FEATURE_VERSION, WINDOW_SIZE, compute_longitudinal_features

app = Celery(
    "hemasight",
    broker=CELERY_BROKER_URL,
)
app.conf.task_serializer = "json"
app.conf.result_serializer = "json"
app.conf.accept_content = ["json"]
app.conf.result_backend = None
app.conf.broker_transport_options = {"confirm_publish": True}
app.conf.task_acks_late = True
app.conf.task_reject_on_worker_lost = True

@lru_cache(maxsize=1)
def _get_session_factory():
    """Initialize the database only when a worker task actually needs it."""
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def load_history_window(db, blood_test: BloodTest) -> List[BloodTest]:
    """Fetch the most recent visits at or before the target in stable order."""
    recent = (
        db.query(BloodTest)
        .filter(
            BloodTest.patient_id == blood_test.patient_id,
            or_(
                BloodTest.date < blood_test.date,
                and_(BloodTest.date == blood_test.date, BloodTest.id <= blood_test.id),
            ),
        )
        .order_by(desc(BloodTest.date), desc(BloodTest.id))
        .limit(WINDOW_SIZE)
        .all()
    )
    return list(reversed(recent))


def compute_features_for_blood_test(db, blood_test: BloodTest, history: List[BloodTest]) -> Feature:
    """Build a v2 feature row; ``db`` is retained for caller compatibility."""
    return Feature(
        patient_id=blood_test.patient_id,
        blood_test_id=blood_test.id,
        feature_version=FEATURE_VERSION,
        **compute_longitudinal_features(blood_test, history),
    )


@app.task(bind=True, autoretry_for=(OperationalError,), retry_backoff=True, retry_kwargs={"max_retries": 5})
def process_blood_test(self, blood_test_id: int) -> dict:
    """Load blood test, compute features, write to features table."""
    db = _get_session_factory()()
    try:
        blood_test = db.query(BloodTest).filter(BloodTest.id == blood_test_id).with_for_update().first()
        if not blood_test:
            return {"blood_test_id": blood_test_id, "status": "not_found"}
        feat = db.query(Feature).filter_by(blood_test_id=blood_test_id, feature_version=FEATURE_VERSION).first()
        if feat is None:
            history = load_history_window(db, blood_test)
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
    db = _get_session_factory()()
    try:
        feat = db.query(Feature).filter(Feature.id == feature_id).with_for_update().first()
        if not feat:
            return {"feature_id": feature_id, "status": "not_found"}
        result = compute_risk_for_feature_id(feature_id)
        if result is None:
            return {"feature_id": feature_id, "status": "not_found"}
        existing = db.query(RiskScore).filter_by(feature_id=feature_id, model_version=result["model_version"]).first()
        if existing:
            return {"feature_id": feature_id, "risk_score_id": existing.id, "status": "ok"}
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
    from hemasight.ml.anomaly import ANOMALY_MODEL_PATH, artifact_version, compute_anomaly_for_feature_id
    if not ANOMALY_MODEL_PATH.exists():
        return {"feature_id": feature_id, "status": "skipped", "reason": "anomaly_model_not_trained"}
    db = _get_session_factory()()
    try:
        feat = db.query(Feature).filter(Feature.id == feature_id).with_for_update().first()
        if not feat:
            return {"feature_id": feature_id, "status": "not_found"}
        result = compute_anomaly_for_feature_id(feature_id)
        if result is None:
            return {"feature_id": feature_id, "status": "not_found"}
        score, is_anomaly = result
        version = artifact_version()
        existing = db.query(AnomalyScore).filter_by(feature_id=feature_id, model_version=version).first()
        if existing:
            return {"feature_id": feature_id, "anomaly_score_id": existing.id, "status": "ok"}
        anomaly_row = AnomalyScore(
            patient_id=feat.patient_id,
            feature_id=feat.id,
            blood_test_id=feat.blood_test_id,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            model_version=version,
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

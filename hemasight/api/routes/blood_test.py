"""Blood test ingestion with durable delivery and retry deduplication."""
from datetime import datetime
import hashlib
import logging

from fastapi import APIRouter, Header, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from hemasight.api.schemas.blood_test import BloodTestCreate, BloodTestResponse
from hemasight.data_pipeline.outbox import dispatch_pending
from hemasight.db.models import BloodTest, IngestionEvent, Patient, get_engine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/blood-test", tags=["blood-test"])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@router.post(
    "",
    response_model=BloodTestResponse,
    status_code=202,
    summary="Ingest blood test",
    response_description="Blood test accepted for processing (202 Accepted).",
)
def post_blood_test(
    payload: BloodTestCreate,
    idempotency_key: str | None = Header(default=None, min_length=1, max_length=128),
):
    """202 means durably saved, even during a queue outage.

    Reuse an Idempotency-Key when retrying a request. Different payloads using
    the same key return 409. Without a key, each POST creates a new test.
    """
    if idempotency_key is not None and not idempotency_key.strip():
        raise HTTPException(status_code=422, detail="Idempotency key must not be blank")
    digest = hashlib.sha256(payload.model_dump_json().encode()).hexdigest()
    with SessionLocal() as db:
        # Retry one insert race on a patient or a shared idempotency key.
        for attempt in range(2):
            try:
                if idempotency_key:
                    existing = db.query(IngestionEvent).filter_by(idempotency_key=idempotency_key).first()
                    if existing:
                        if existing.request_sha256 != digest:
                            raise HTTPException(status_code=409, detail="Idempotency key was already used for a different blood test")
                        return BloodTestResponse(blood_test_id=existing.blood_test_id, patient_id=payload.patient_id)
                patient = db.query(Patient).filter_by(external_id=payload.patient_id).first()
                if not patient:
                    patient = Patient(external_id=payload.patient_id)
                    db.add(patient)
                    db.flush()
                bt = BloodTest(
                    patient_id=patient.id,
                    date=datetime.combine(payload.date, datetime.min.time()),
                    **payload.model_dump(exclude={"patient_id", "date"}),
                )
                db.add(bt)
                db.flush()
                event = IngestionEvent(
                    blood_test_id=bt.id, idempotency_key=idempotency_key, request_sha256=digest,
                )
                db.add(event)
                db.flush()
                blood_test_id, event_id = bt.id, event.id
                db.commit()
                break
            except HTTPException:
                raise
            except IntegrityError:
                db.rollback()
                if attempt:
                    raise HTTPException(status_code=503, detail="Unable to save blood test; retry with the same idempotency key") from None
            except Exception:
                db.rollback()
                logger.error("Blood test persistence failed")
                raise HTTPException(status_code=503, detail="Unable to save blood test; retry with the same idempotency key") from None
    try:
        dispatch_pending(session_factory=SessionLocal, event_id=event_id)
    except Exception:
        logger.warning("Blood test saved; queue delivery deferred to dispatcher")
    return BloodTestResponse(blood_test_id=blood_test_id, patient_id=payload.patient_id)

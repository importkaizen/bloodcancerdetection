"""Blood test ingestion route."""
from datetime import datetime

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import sessionmaker

from hemasight.api.schemas.blood_test import BloodTestCreate, BloodTestResponse
from hemasight.data_pipeline.producer import publish_blood_test_ingested
from hemasight.db.models import BloodTest, Patient, get_engine

router = APIRouter(prefix="/blood-test", tags=["blood-test"])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


@router.post(
    "",
    response_model=BloodTestResponse,
    status_code=202,
    summary="Ingest blood test",
    response_description="Blood test accepted for processing (202 Accepted).",
)
def post_blood_test(payload: BloodTestCreate):
    """Accept a blood test, persist to DB, and publish to queue for feature processing."""
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.external_id == payload.patient_id).first()
        if not patient:
            patient = Patient(external_id=payload.patient_id)
            db.add(patient)
            db.commit()
            db.refresh(patient)
        bt = BloodTest(
            patient_id=patient.id,
            date=datetime.combine(payload.date, datetime.min.time()),
            wbc=payload.wbc,
            rbc=payload.rbc,
            platelets=payload.platelets,
            hemoglobin=payload.hemoglobin,
            lymphocytes=payload.lymphocytes,
        )
        db.add(bt)
        db.commit()
        db.refresh(bt)
        msg_id = publish_blood_test_ingested(bt.id, payload.patient_id)
        return BloodTestResponse(
            blood_test_id=bt.id,
            patient_id=payload.patient_id,
            message="Blood test accepted for processing",
            status="accepted",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

"""Patients and dashboard API: list patients, blood tests, risk scores."""
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker

from hemasight.db.models import BloodTest, Patient, RiskScore, get_engine

router = APIRouter(tags=["patients"])
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


class BloodTestOut(BaseModel):
    id: int
    patient_id: int
    date: datetime
    wbc: Optional[float]
    rbc: Optional[float]
    platelets: Optional[float]
    hemoglobin: Optional[float]
    lymphocytes: Optional[float]
    created_at: datetime


class RiskScoreOut(BaseModel):
    id: int
    patient_id: int
    score: float
    level: str
    model_version: str
    message: Optional[str]
    computed_at: datetime


class PatientOut(BaseModel):
    id: int
    external_id: str
    created_at: datetime


@router.get("/patients", response_model=List[PatientOut])
def list_patients():
    """List all patients."""
    db = SessionLocal()
    try:
        patients = db.query(Patient).order_by(Patient.id).all()
        return [PatientOut(id=p.id, external_id=p.external_id, created_at=p.created_at) for p in patients]
    finally:
        db.close()


@router.get("/patients/{patient_id}/blood-tests", response_model=List[BloodTestOut])
def get_patient_blood_tests(patient_id: int):
    """Get blood tests for a patient (by internal id)."""
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        tests = db.query(BloodTest).filter(BloodTest.patient_id == patient_id).order_by(BloodTest.date).all()
        return [BloodTestOut(id=t.id, patient_id=t.patient_id, date=t.date, wbc=t.wbc, rbc=t.rbc, platelets=t.platelets, hemoglobin=t.hemoglobin, lymphocytes=t.lymphocytes, created_at=t.created_at) for t in tests]
    finally:
        db.close()


@router.get("/patients/{patient_id}/risk-scores", response_model=List[RiskScoreOut])
def get_patient_risk_scores(patient_id: int):
    """Get risk scores for a patient."""
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        scores = db.query(RiskScore).filter(RiskScore.patient_id == patient_id).order_by(RiskScore.computed_at).all()
        return [RiskScoreOut(id=s.id, patient_id=s.patient_id, score=s.score, level=s.level, model_version=s.model_version, message=s.message, computed_at=s.computed_at) for s in scores]
    finally:
        db.close()

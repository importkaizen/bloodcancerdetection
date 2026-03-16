"""Pydantic schemas for blood test API."""
from datetime import date as date_type, datetime
from typing import Optional

from pydantic import BaseModel, Field


class BloodTestCreate(BaseModel):
    """Request body for POST /blood-test."""

    patient_id: str = Field(..., description="External patient identifier")
    date: date_type = Field(..., description="Date of the blood test")
    wbc: Optional[float] = Field(None, ge=0, le=100, description="White blood cells (K/uL)")
    rbc: Optional[float] = Field(None, ge=0, le=10, description="Red blood cells (M/uL)")
    platelets: Optional[float] = Field(None, ge=0, le=2000, description="Platelets (K/uL)")
    hemoglobin: Optional[float] = Field(None, ge=0, le=25, description="Hemoglobin (g/dL)")
    lymphocytes: Optional[float] = Field(None, ge=0, le=100, description="Lymphocytes (%)")

    model_config = {"json_schema_extra": {"example": {"patient_id": "123", "date": "2025-05-01", "wbc": 7.2, "rbc": 4.8, "platelets": 210, "hemoglobin": 13.5, "lymphocytes": 40}}}


class BloodTestResponse(BaseModel):
    """202 response after accepting a blood test."""

    blood_test_id: int = Field(..., description="ID of the created blood test record")
    patient_id: str = Field(..., description="Patient external ID")
    message: str = Field(default="Blood test accepted for processing")
    status: str = Field(default="accepted")

    model_config = {"json_schema_extra": {"example": {"blood_test_id": 1, "patient_id": "123", "message": "Blood test accepted for processing", "status": "accepted"}}}

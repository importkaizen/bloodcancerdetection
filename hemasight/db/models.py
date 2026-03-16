"""SQLAlchemy models for HemaSight. PostgreSQL + TimescaleDB hypertables."""
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    metadata_ = Column("metadata", JSON, nullable=True)

    blood_tests = relationship("BloodTest", back_populates="patient")
    features = relationship("Feature", back_populates="patient")
    risk_scores = relationship("RiskScore", back_populates="patient")


class BloodTest(Base):
    __tablename__ = "blood_tests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(DateTime(timezone=True), nullable=False)
    wbc = Column(Float, nullable=True)  # White blood cells
    rbc = Column(Float, nullable=True)  # Red blood cells
    platelets = Column(Float, nullable=True)
    hemoglobin = Column(Float, nullable=True)
    lymphocytes = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    patient = relationship("Patient", back_populates="blood_tests")
    features = relationship("Feature", back_populates="blood_test")


class Feature(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    blood_test_id = Column(Integer, ForeignKey("blood_tests.id", ondelete="CASCADE"), nullable=False, index=True)
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    feature_version = Column(String(32), nullable=False, default="v1")

    # Derived features
    wbc_trend = Column(Float, nullable=True)
    platelet_var = Column(Float, nullable=True)
    hemoglobin_drop_rate = Column(Float, nullable=True)
    lymphocyte_spike = Column(Float, nullable=True)
    wbc_rolling_avg = Column(Float, nullable=True)
    rbc_rolling_avg = Column(Float, nullable=True)
    platelets_rolling_avg = Column(Float, nullable=True)
    hemoglobin_rolling_avg = Column(Float, nullable=True)
    # Raw values at time of feature computation (for model input)
    wbc = Column(Float, nullable=True)
    rbc = Column(Float, nullable=True)
    platelets = Column(Float, nullable=True)
    hemoglobin = Column(Float, nullable=True)
    lymphocytes = Column(Float, nullable=True)

    patient = relationship("Patient", back_populates="features")
    blood_test = relationship("BloodTest", back_populates="features")


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_id = Column(Integer, ForeignKey("features.id", ondelete="SET NULL"), nullable=True, index=True)
    blood_test_id = Column(Integer, ForeignKey("blood_tests.id", ondelete="SET NULL"), nullable=True, index=True)
    score = Column(Float, nullable=False)  # 0.0 - 1.0
    level = Column(String(32), nullable=False)  # LOW, MEDIUM, ELEVATED
    model_version = Column(String(64), nullable=False, default="v1")
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    message = Column(Text, nullable=True)  # e.g. "unusual drift that may indicate..."

    patient = relationship("Patient", back_populates="risk_scores")


class AnomalyScore(Base):
    __tablename__ = "anomaly_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_id = Column(Integer, ForeignKey("features.id", ondelete="SET NULL"), nullable=True, index=True)
    blood_test_id = Column(Integer, ForeignKey("blood_tests.id", ondelete="SET NULL"), nullable=True, index=True)
    anomaly_score = Column(Float, nullable=False)
    is_anomaly = Column(Integer, nullable=False, default=0)  # 0/1 for SQLite compat
    model_version = Column(String(64), nullable=False, default="isolation_forest_v1")
    computed_at = Column(DateTime(timezone=True), default=datetime.utcnow)


def get_engine(url: Optional[str] = None):
    from hemasight.config import DATABASE_URL
    return create_engine(url or DATABASE_URL, pool_pre_ping=True)


def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    # Convert to TimescaleDB hypertables (run after first create_all)
    with engine.connect() as conn:
        for table, time_col in [("blood_tests", "created_at"), ("features", "computed_at"), ("risk_scores", "computed_at")]:
            try:
                conn.execute(text(f"SELECT create_hypertable('{table}', '{time_col}', if_not_exists => TRUE);"))
                conn.commit()
            except Exception:
                conn.rollback()

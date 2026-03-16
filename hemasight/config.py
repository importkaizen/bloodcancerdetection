"""Central config from environment."""
import os
from pathlib import Path

# HemaSight package root (directory containing this file)
HEMASIGHT_ROOT = Path(__file__).resolve().parent

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://hemasight:hemasight@localhost:5432/hemasight",
)

# RabbitMQ
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
BLOOD_TEST_QUEUE = os.getenv("BLOOD_TEST_QUEUE", "blood_test.ingested")

# Celery
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", RABBITMQ_URL)

# ML
ML_MODELS_DIR = Path(os.getenv("ML_MODELS_DIR", str(HEMASIGHT_ROOT / "ml" / "models")))
RISK_MODEL_PATH = ML_MODELS_DIR / "risk_model.pkl"
SCALER_PATH = ML_MODELS_DIR / "scaler.pkl"
MODEL_CONFIG_PATH = ML_MODELS_DIR / "config.json"

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

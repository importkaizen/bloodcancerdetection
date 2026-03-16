"""Anomaly detection: Isolation Forest and optional Autoencoder."""
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from hemasight.config import ML_MODELS_DIR

FEATURE_COLS = [
    "wbc", "rbc", "platelets", "hemoglobin", "lymphocytes",
    "wbc_trend", "platelet_var", "hemoglobin_drop_rate", "lymphocyte_spike",
    "wbc_rolling_avg", "rbc_rolling_avg", "platelets_rolling_avg", "hemoglobin_rolling_avg",
]
ANOMALY_MODEL_PATH = ML_MODELS_DIR / "anomaly_isolation_forest.pkl"
ANOMALY_CONFIG_PATH = ML_MODELS_DIR / "anomaly_config.json"
VERSION = "isolation_forest_v1"


def feature_row_to_vector(feature_row: dict) -> np.ndarray:
    vec = []
    for c in FEATURE_COLS:
        v = feature_row.get(c)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        vec.append(float(v))
    return np.array(vec).reshape(1, -1)


def fit_isolation_forest(X: np.ndarray, contamination: float = 0.1, random_state: int = 42) -> Path:
    """Fit Isolation Forest on feature matrix X (n_samples, n_features). Save model and config."""
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X)
    import joblib
    joblib.dump(clf, ANOMALY_MODEL_PATH)
    config = {"feature_columns": FEATURE_COLS, "version": VERSION}
    with open(ANOMALY_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return ANOMALY_MODEL_PATH


def predict_anomaly(feature_vector: np.ndarray, model=None) -> Tuple[float, int]:
    """
    Return (anomaly_score, is_anomaly).
    anomaly_score: higher = more anomalous. We use negative of decision_function so higher = more anomalous.
    is_anomaly: 1 if predicted anomaly, 0 otherwise.
    """
    if model is None:
        import joblib
        model = joblib.load(ANOMALY_MODEL_PATH)
    pred = model.predict(feature_vector)[0]  # -1 or 1
    score = -model.decision_function(feature_vector)[0]  # higher = more anomalous
    is_anomaly = 1 if pred == -1 else 0
    return float(score), is_anomaly


def compute_anomaly_for_feature_id(feature_id: int) -> Optional[Tuple[float, int]]:
    """Load feature from DB, run anomaly model, return (anomaly_score, is_anomaly) or None."""
    if not ANOMALY_MODEL_PATH.exists():
        return None
    from hemasight.db.models import Feature, get_engine
    from sqlalchemy.orm import sessionmaker
    import joblib
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        row = db.query(Feature).filter(Feature.id == feature_id).first()
        if not row:
            return None
        row_dict = {c.key: getattr(row, c.key) for c in row.__table__.columns}
        vec = feature_row_to_vector(row_dict)
        model = joblib.load(ANOMALY_MODEL_PATH)
        return predict_anomaly(vec, model=model)
    finally:
        db.close()


# --- Optional Autoencoder (PyTorch) for subtle pattern changes ---
AUTOENCODER_PATH = ML_MODELS_DIR / "anomaly_autoencoder.pt"
AUTOENCODER_SCALER_PATH = ML_MODELS_DIR / "anomaly_autoencoder_scaler.pkl"


def _autoencoder_model(n_features: int = 13, latent: int = 4):
    """Simple MLP autoencoder."""
    import torch
    import torch.nn as nn

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 8),
                nn.ReLU(),
                nn.Linear(8, latent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 8),
                nn.ReLU(),
                nn.Linear(8, n_features),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    return Autoencoder()


def fit_autoencoder(X: np.ndarray, epochs: int = 50, latent: int = 4) -> Path:
    """Train autoencoder on normal feature data. Reconstruction error used as anomaly score."""
    import torch
    from sklearn.preprocessing import StandardScaler
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_features = X_scaled.shape[1]
    model = _autoencoder_model(n_features=n_features, latent=latent)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        recon, _ = model(X_t)
        loss = torch.nn.functional.mse_loss(recon, X_t)
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.save({"state_dict": model.state_dict(), "n_features": n_features, "latent": latent}, AUTOENCODER_PATH)
    import joblib
    joblib.dump(scaler, AUTOENCODER_SCALER_PATH)
    return AUTOENCODER_PATH


def predict_anomaly_autoencoder(feature_vector: np.ndarray, model=None, scaler=None) -> float:
    """Reconstruction error as anomaly score (higher = more anomalous)."""
    import torch
    if model is None or scaler is None:
        import joblib
        if not AUTOENCODER_PATH.exists():
            return 0.0
        ck = torch.load(AUTOENCODER_PATH, map_location="cpu")
        model = _autoencoder_model(n_features=ck["n_features"], latent=ck["latent"])
        model.load_state_dict(ck["state_dict"])
        scaler = joblib.load(AUTOENCODER_SCALER_PATH)
    X = scaler.transform(feature_vector)
    with torch.no_grad():
        recon, _ = model(torch.tensor(X, dtype=torch.float32))
        err = ((recon.numpy() - X) ** 2).mean(axis=1)[0]
    return float(err)

"""Anomaly detection: Isolation Forest and optional Autoencoder."""
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from hemasight.config import ML_MODELS_DIR

FEATURE_COLS = [
    "wbc", "rbc", "platelets", "hemoglobin", "lymphocytes",
    "wbc_trend", "platelet_var", "hemoglobin_drop_rate", "lymphocyte_spike",
    "wbc_rolling_avg", "rbc_rolling_avg", "platelets_rolling_avg", "hemoglobin_rolling_avg",
]
ANOMALY_MODEL_PATH = ML_MODELS_DIR / "anomaly_isolation_forest.pkl"
ANOMALY_CONFIG_PATH = ML_MODELS_DIR / "anomaly_config.json"
VERSION = "isolation_forest_v2"


def feature_row_to_vector(feature_row: dict) -> np.ndarray:
    from hemasight.ml.inference import feature_row_to_vector as ordered_vector
    return ordered_vector(feature_row, FEATURE_COLS)


def artifact_version() -> str:
    if ANOMALY_CONFIG_PATH.exists():
        return json.loads(ANOMALY_CONFIG_PATH.read_text())["version"]
    return "isolation_forest_v1"


def fit_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.1,
    random_state: int = 42,
    *,
    feature_version: str = "v1",
) -> Path:
    """Fit and save Isolation Forest with its input feature version.

    Pass ``feature_version="v2"`` only when X uses the corrected elapsed-day,
    recent-window features. The default preserves existing v1 training callers.
    """
    if not isinstance(feature_version, str) or not feature_version.strip():
        raise ValueError("feature_version must be a nonempty string")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or X.shape[1] != len(FEATURE_COLS):
        raise ValueError("Training matrix must be nonempty with 13 ordered features")
    if np.isinf(X).any() or np.isnan(X).all(axis=0).any():
        raise ValueError("Training features must not be infinite or entirely missing")
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", IsolationForest(contamination=contamination, random_state=random_state)),
    ])
    clf.fit(X)
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(clf, ANOMALY_MODEL_PATH)
    config = {
        "feature_columns": FEATURE_COLS,
        "version": VERSION,
        "feature_version": feature_version,
        "preprocessing": "training_fitted_median_imputer",
    }
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
    feature_vector = np.asarray(feature_vector, dtype=float)
    if feature_vector.shape != (1, len(FEATURE_COLS)) or np.isinf(feature_vector).any():
        raise ValueError("Expected one row of 13 finite-or-missing features")
    has_imputer = isinstance(model, Pipeline) and any(isinstance(step, SimpleImputer) for _, step in model.steps)
    if np.isnan(feature_vector).any() and not has_imputer:
        raise ValueError("Legacy anomaly model has no trained imputer; retrain before scoring missing features")
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
        config = {}
        if ANOMALY_CONFIG_PATH.exists():
            with open(ANOMALY_CONFIG_PATH) as f:
                config = json.load(f)
        expected_version = config.get("feature_version", "v1")
        if row.feature_version != expected_version:
            raise ValueError(
                f"Anomaly model expects feature version {expected_version!r}, "
                f"but row {feature_id} uses {row.feature_version!r}. "
                "Retrain the Isolation Forest on matching features and save "
                "their feature_version before scoring this row."
            )
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

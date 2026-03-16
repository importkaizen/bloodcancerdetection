"""Optional LSTM model for time-series risk (sequence of feature vectors)."""
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from hemasight.config import ML_MODELS_DIR, MODEL_CONFIG_PATH, RISK_MODEL_PATH, SCALER_PATH

FEATURE_COLUMNS = [
    "wbc", "rbc", "platelets", "hemoglobin", "lymphocytes",
    "wbc_trend", "platelet_var", "hemoglobin_drop_rate", "lymphocyte_spike",
    "wbc_rolling_avg", "rbc_rolling_avg", "platelets_rolling_avg", "hemoglobin_rolling_avg",
]
LSTM_MODEL_PATH = ML_MODELS_DIR / "risk_lstm.pt"
SEQ_LEN_KEY = "seq_len"
HIDDEN_KEY = "hidden_size"


class LSTMRisk(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h.squeeze(0)))


def train_lstm(
    X_seq: np.ndarray,
    y: np.ndarray,
    seq_len: Optional[int] = None,
    hidden_size: int = 32,
    epochs: int = 30,
) -> dict:
    """
    X_seq: (n_samples, seq_len, n_features). If 2D (n_samples, n_features), seq_len=1.
    y: (n_samples,) binary labels.
    """
    if X_seq.ndim == 2:
        X_seq = X_seq[:, None, :]
    n_samples, seq_len_actual, n_features = X_seq.shape
    seq_len = seq_len or seq_len_actual
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.preprocessing import StandardScaler
    import joblib
    X_flat = X_seq.reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(n_samples, seq_len_actual, n_features)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    model = LSTMRisk(n_features, hidden_size=hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        model.train()
        pred = model(X_t)
        loss = nn.functional.binary_cross_entropy(pred, y_t)
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.save({"state_dict": model.state_dict(), "n_features": n_features, "hidden_size": hidden_size, "seq_len": seq_len_actual}, LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    config = {"feature_columns": FEATURE_COLUMNS, "model_version": "lstm_v1", "model_type": "lstm", "thresholds": {"LOW": 0.33, "MEDIUM": 0.66}, "seq_len": seq_len_actual}
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return {"model_version": "lstm_v1"}


def predict_lstm(feature_vector: np.ndarray, model=None, scaler=None, config=None) -> float:
    """feature_vector: (1, n_features) or (1, seq_len, n_features). Returns risk score in [0,1]."""
    import joblib
    if model is None or scaler is None or config is None:
        ck = torch.load(LSTM_MODEL_PATH, map_location="cpu")
        model = LSTMRisk(ck["n_features"], hidden_size=ck["hidden_size"])
        model.load_state_dict(ck["state_dict"])
        scaler = joblib.load(SCALER_PATH)
        with open(MODEL_CONFIG_PATH) as f:
            config = json.load(f)
    X = feature_vector
    if X.ndim == 2:
        X = X[:, None, :]
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    with torch.no_grad():
        score = model(torch.tensor(X, dtype=torch.float32)).item()
    return float(score)

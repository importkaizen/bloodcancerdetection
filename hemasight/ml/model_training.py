"""Risk model training: RF/XGBoost on CBC-derived features. Saves model, scaler, config."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from hemasight.config import ML_MODELS_DIR, MODEL_CONFIG_PATH, RISK_MODEL_PATH, SCALER_PATH

FEATURE_COLUMNS = [
    "wbc",
    "rbc",
    "platelets",
    "hemoglobin",
    "lymphocytes",
    "wbc_trend",
    "platelet_var",
    "hemoglobin_drop_rate",
    "lymphocyte_spike",
    "wbc_rolling_avg",
    "rbc_rolling_avg",
    "platelets_rolling_avg",
    "hemoglobin_rolling_avg",
]
DEFAULT_LABEL_COL = "label"
MODEL_VERSION = "v1"
THRESHOLDS = {"LOW": 0.33, "MEDIUM": 0.66}


def load_training_data(path: str, label_col: str = DEFAULT_LABEL_COL) -> tuple:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    X = df[FEATURE_COLUMNS].copy().fillna(df[FEATURE_COLUMNS].median())
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not in data.")
    y = df[label_col]
    return X, y


def train(
    data_path: str,
    label_col: str = DEFAULT_LABEL_COL,
    model_type: str = "xgboost",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    X, y = load_training_data(data_path, label_col=label_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if model_type == "xgboost":
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=random_state, use_label_encoder=False, eval_metric="logloss")
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, RISK_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    config = {"feature_columns": FEATURE_COLUMNS, "model_version": MODEL_VERSION, "thresholds": THRESHOLDS, "model_type": model_type}
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    return {"classification_report": report, "model_version": MODEL_VERSION}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Path to CSV or Parquet")
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--model-type", choices=["xgboost", "rf"], default="xgboost")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    print(train(args.data_path, label_col=args.label_col, model_type=args.model_type, test_size=args.test_size))

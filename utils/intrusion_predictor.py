import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import INTRUSION_FEATURES, safe_numeric

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "intrusion_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"


def load_intrusion_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Intrusion model not found.")
    artifact = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH) if FEATURE_COLUMNS_PATH.exists() else INTRUSION_FEATURES
    if isinstance(artifact, dict):
        model = artifact.get("model")
        scaler = artifact.get("scaler", scaler)
        feature_columns = artifact.get("feature_columns", feature_columns)
    else:
        model = artifact
    if model is None:
        raise FileNotFoundError("Intrusion model could not be loaded.")
    return {"model": model, "scaler": scaler, "feature_columns": list(feature_columns)}


def validate_intrusion_input(data):
    cleaned = {}
    for field in INTRUSION_FEATURES:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        value = safe_numeric(data[field])
        if pd.isna(value):
            raise ValueError(f"Please provide a valid numeric value for {field}.")
        cleaned[field] = float(value)
    return cleaned


def predict_intrusion_record(data):
    artifact = load_intrusion_artifacts()
    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]

    row = {column: float(data.get(column, 0)) for column in feature_columns}
    frame = pd.DataFrame([row], columns=feature_columns)
    if scaler is not None:
        transformed = scaler.transform(frame)
    else:
        transformed = frame.to_numpy()

    probabilities = model.predict_proba(transformed)[0]
    predicted_index = int(np.argmax(probabilities))
    label = "INTRUSION" if predicted_index == 1 else "NORMAL"
    confidence = float(probabilities[predicted_index] * 100)

    if confidence >= 80:
        risk = "HIGH"
    elif confidence >= 60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    explanation = (
        "The model classified this traffic as potentially malicious based on the supplied network-flow characteristics. "
        "This result is an ML-based prediction and should be used as a support signal rather than absolute certainty."
        if label == "INTRUSION"
        else "The model classified this traffic as normal based on the supplied network-flow characteristics."
    )
    return label, confidence, risk, explanation

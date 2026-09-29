import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INTRUSION_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Average Packet Size",
    "Subflow Fwd Bytes",
]

DATASET_INFO = {
    "intrusion_dataset": "CIC-IDS-2017 (recommended)",
    "intrusion_source": "Canadian Institute for Cybersecurity / CICIDS2017",
    "mail_dataset": "SMS Spam Collection or Enron spam corpus (recommended for real-world training)",
    "model_used": "Random Forest for intrusion detection; TF-IDF + Logistic Regression for mail guard",
    "features": INTRUSION_FEATURES,
    "preprocessing": [
        "Missing-value handling with median imputation",
        "Infinite-value removal and replacement",
        "Scaling using StandardScaler for numerical model input",
        "Consistent feature ordering during training and inference",
    ],
    "target_labels": [
        "NORMAL",
        "INTRUSION",
    ],
}


def safe_numeric(value):
    if value is None or value == "":
        return np.nan
    try:
        converted = float(value)
        if math.isfinite(converted):
            return converted
        return np.nan
    except (TypeError, ValueError):
        return np.nan


def clean_intrusion_frame(frame):
    df = frame.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for column in df.columns:
        if df[column].dtype.kind in "ifub":
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.fillna(df.median(numeric_only=True))
    return df


def select_intrusion_columns(frame):
    target_candidates = ["Label", "label", "Class", "class", "Attack", "attack", "Target", "target"]
    target_column = next((name for name in target_candidates if name in frame.columns), None)
    if target_column is None:
        if "Label" not in frame.columns and "label" not in frame.columns:
            raise ValueError("The intrusion dataset does not contain a target label column.")
        target_column = "Label"
    columns = [column for column in frame.columns if column != target_column]
    numeric_columns = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    return target_column, numeric_columns

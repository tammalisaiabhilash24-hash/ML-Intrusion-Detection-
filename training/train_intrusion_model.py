import sys
import urllib.request
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.preprocessing import INTRUSION_FEATURES, clean_intrusion_frame

DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
INTRUSION_DATASET_URLS = {
    "Monday-WorkingHours.pcap_ISCX.csv": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Monday-WorkingHours.pcap_ISCX.csv?download=1",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Tuesday-WorkingHours.pcap_ISCX.csv?download=1",
}

warnings.filterwarnings("ignore")


def download_intrusion_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []
    for filename, url in INTRUSION_DATASET_URLS.items():
        target_path = DATA_DIR / filename
        if not target_path.exists():
            print(f"Downloading {filename} from the official public CIC-IDS-2017 mirror...")
            urllib.request.urlretrieve(url, target_path)
        downloaded_paths.append(target_path)
    return downloaded_paths


def find_intrusion_dataset_files():
    paths = [
        DATA_DIR / filename for filename in INTRUSION_DATASET_URLS
        if (DATA_DIR / filename).exists()
    ]
    if paths:
        return paths
    return download_intrusion_dataset()


def generate_demo_intrusion_dataset():
    raise RuntimeError(
        "Demo intrusion data fallback is disabled. Real public CIC-IDS-2017 data is required. "
        "Please place or download the required CSV files into data/raw/."
    )


def prepare_intrusion_data(df):
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    if "Label" not in df.columns:
        raise ValueError("The intrusion dataset must include a 'Label' column.")

    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    label_map = {
        "NORMAL": 0,
        "BENIGN": 0,
        "SAFE": 0,
        "ATTACK": 1,
        "MALICIOUS": 1,
        "INTRUSION": 1,
        "ANOMALY": 1,
        "FTP-PATATOR": 1,
        "SSH-PATATOR": 1,
        "0": 0,
        "1": 1,
    }
    df["Label"] = df["Label"].map(label_map)
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(int)

    feature_columns = [column for column in INTRUSION_FEATURES if column in df.columns]
    if not feature_columns:
        raise ValueError("The real intrusion dataset does not include the expected feature columns required by this project.")

    df = df[feature_columns + ["Label"]]
    df = clean_intrusion_frame(df)
    return df


def train_intrusion_model():
    dataset_paths = find_intrusion_dataset_files()
    if not dataset_paths:
        raise FileNotFoundError(
            "Intrusion training data is missing. Download the real CIC-IDS-2017 CSV files into data/raw/ or run the automatic download step."
        )

    frames = []
    for path in dataset_paths:
        print(f"Loading intrusion dataset from: {path}")
        frames.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} real CIC-IDS-2017 rows.")

    df = prepare_intrusion_data(df)

    target_column = "Label"
    feature_columns = [column for column in INTRUSION_FEATURES if column in df.columns]
    X = df[feature_columns]
    y = df[target_column].astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("\nRandom Forest metrics:")
    print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    print(f"Precision: {precision_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, rf_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, rf_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_pred))

    joblib.dump({"model": rf_model, "scaler": scaler, "feature_columns": feature_columns}, MODEL_DIR / "intrusion_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")
    print("\nReal model saved successfully to models/intrusion_model.pkl")


if __name__ == "__main__":
    train_intrusion_model()

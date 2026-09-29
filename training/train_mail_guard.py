import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"
MAIL_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

warnings.filterwarnings("ignore")


def download_mail_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "smsspamcollection.zip"
    if not zip_path.exists():
        print("Downloading SMS Spam Collection from the UCI Machine Learning Repository...")
        urllib.request.urlretrieve(MAIL_DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DATA_DIR)

    extracted_files = [
        DATA_DIR / "SMSSpamCollection",
        DATA_DIR / "SMSSpamCollection.txt",
        DATA_DIR / "smsspamcollection",
        DATA_DIR / "spam.csv",
    ]
    for path in extracted_files:
        if path.exists():
            return path

    raise FileNotFoundError("The SMS Spam Collection file was not found after extraction. Please inspect data/raw/.")


def find_mail_dataset():
    candidates = [
        DATA_DIR / "SMSSpamCollection",
        DATA_DIR / "SMSSpamCollection.txt",
        DATA_DIR / "smsspamcollection",
        DATA_DIR / "spam.csv",
        DATA_DIR / "mail.csv",
        DATA_DIR / "mail_dataset.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in sorted(DATA_DIR.iterdir()):
        if path.suffix.lower() == ".csv" and ("spam" in path.name.lower() or "mail" in path.name.lower()):
            return path

    return download_mail_dataset()


def generate_demo_mail_dataset():
    raise RuntimeError(
        "Demo mail data fallback is disabled. Real public SMS Spam Collection data is required. "
        "Please place or download the real dataset into data/raw/."
    )


def load_mail_dataset():
    dataset_path = find_mail_dataset()
    print(f"Loading mail dataset from: {dataset_path}")
    if dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path, low_memory=False)
        if "label" in df.columns and "text" in df.columns:
            return df[["text", "label"]].copy()
        if "message" in df.columns and "label" in df.columns:
            return df[["message", "label"]].rename(columns={"message": "text"})
        raise ValueError("The CSV mail dataset is missing the required text and label columns.")

    if dataset_path.name.lower() in {"smsspamcollection", "smsspamcollection.txt"}:
        df = pd.read_csv(dataset_path, sep="\t", header=None, names=["label", "text"], encoding="utf-8", engine="python")
        return df[["text", "label"]].copy()

    raise ValueError(f"Unsupported mail dataset file format: {dataset_path}")


def train_mail_model():
    dataset_path = find_mail_dataset()
    if dataset_path is None:
        raise FileNotFoundError("Mail training data is missing. Download the UCI SMS Spam Collection into data/raw/.")

    df = load_mail_dataset()
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].fillna("").astype(str)

    valid_labels = {"safe": 0, "ham": 0, "legit": 0, "normal": 0, "spam": 1, "phishing": 1, "suspicious": 1}
    df["label_code"] = df["label"].map(valid_labels)
    df = df.dropna(subset=["label_code"]).reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_code"], test_size=0.2, random_state=42, stratify=df["label_code"]
    )

    model_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)

    print("\nMail Guard evaluation metrics:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    artifact = {
        "model": model_pipeline,
        "label_mapping": {0: "SAFE", 1: "SPAM-SUSPICIOUS"},
        "description": "Binary text classifier trained on the UCI SMS Spam Collection."
    }
    joblib.dump(artifact, MODEL_DIR / "mail_guard_model.pkl")
    print("\nReal Mail Guard model saved successfully to models/mail_guard_model.pkl")


if __name__ == "__main__":
    train_mail_model()

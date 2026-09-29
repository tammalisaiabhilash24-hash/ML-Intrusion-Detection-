import json
from pathlib import Path

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def display_model_status():
    model_files = sorted(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        print("No model files have been created yet.")
        return

    print("Available model files:")
    for path in model_files:
        print(f"- {path.name}")

    intrusion_model = MODELS_DIR / "intrusion_model.pkl"
    mail_model = MODELS_DIR / "mail_guard_model.pkl"

    if intrusion_model.exists():
        artifact = joblib.load(intrusion_model)
        print(f"Intrusion model type: {type(artifact)}")
    if mail_model.exists():
        artifact = joblib.load(mail_model)
        print(f"Mail model type: {type(artifact)}")


if __name__ == "__main__":
    display_model_status()

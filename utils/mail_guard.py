import re
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "mail_guard_model.pkl"


def load_mail_guard_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Mail Guard model not found.")
    return joblib.load(MODEL_PATH)


def detect_email_indicators(sender, subject, message, url=""):
    text = f"{sender} {subject} {message} {url}".lower()
    indicators = []

    if re.search(r"https?://|www\.|\.com|\.net|\.org", text):
        indicators.append("suspicious URL")
    if any(term in text for term in ["urgent", "immediately", "act now", "today only", "limited time"]):
        indicators.append("urgent language")
    if any(term in text for term in ["password", "login now", "verify account", "update credentials", "otp", "one-time password"]):
        indicators.append("credential request")
    if any(term in text for term in ["payment", "invoice", "wire transfer", "bank details", "send money", "refund", "transfer funds"]):
        indicators.append("financial request")
    if any(term in text for term in ["attachment", "invoice attached", "document attached", "click below", "open attachment"]):
        indicators.append("attachment mention")

    if not indicators:
        return ["no major indicators detected"]
    return indicators


def predict_mail_text(sender, subject, message, url=""):
    artifact = load_mail_guard_model()
    if isinstance(artifact, dict):
        model = artifact.get("model")
        label_mapping = artifact.get("label_mapping", {0: "SAFE", 1: "SPAM-SUSPICIOUS"})
    else:
        model = artifact
        label_mapping = {0: "SAFE", 1: "SPAM-SUSPICIOUS"}

    text = f"{subject} {message} {url}".strip()
    if not text:
        raise ValueError("Email content cannot be empty.")

    probabilities = model.predict_proba([text])[0]
    predicted_index = int(probabilities.argmax())
    label = label_mapping.get(predicted_index, "SAFE")

    if label == "SPAM-SUSPICIOUS":
        risk_level = "HIGH" if probabilities[predicted_index] > 0.8 else "MEDIUM"
    else:
        risk_level = "LOW" if probabilities[predicted_index] > 0.6 else "MEDIUM"

    confidence = float(probabilities[predicted_index] * 100)
    if confidence >= 80:
        risk = "HIGH"
    elif confidence >= 60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    indicators = detect_email_indicators(sender, subject, message, url)
    if label == "SAFE":
        explanation = "The message content appears safe based on the trained text classifier and the observed indicators."
    else:
        explanation = "The classifier detected suspicious language patterns and related risk indicators. This result is not absolute proof of malicious activity, but it is a useful warning signal."
    return label, confidence, risk, explanation, indicators

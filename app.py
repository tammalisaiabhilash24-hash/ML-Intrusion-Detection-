import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request

from utils.intrusion_predictor import predict_intrusion_record, validate_intrusion_input
from utils.mail_guard import detect_email_indicators, predict_mail_text

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", DATABASE_DIR / "app.db"))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = get_db_connection()
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS intrusion_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            details TEXT,
            input_data TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS mail_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sender TEXT,
            subject TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            indicators TEXT,
            details TEXT,
            message TEXT
        )
        """
    )
    connection.commit()
    connection.close()


def save_intrusion_prediction(prediction, confidence, risk_level, details, input_data):
    connection = get_db_connection()
    connection.execute(
        """
        INSERT INTO intrusion_predictions (timestamp, prediction, confidence, risk_level, details, input_data)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prediction,
            float(confidence),
            risk_level,
            details,
            json.dumps(input_data, ensure_ascii=False),
        ),
    )
    connection.commit()
    connection.close()


def save_mail_prediction(sender, subject, prediction, confidence, risk_level, indicators, details, message):
    connection = get_db_connection()
    connection.execute(
        """
        INSERT INTO mail_predictions (timestamp, sender, subject, prediction, confidence, risk_level, indicators, details, message)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            sender,
            subject,
            prediction,
            float(confidence),
            risk_level,
            json.dumps(indicators, ensure_ascii=False),
            details,
            message,
        ),
    )
    connection.commit()
    connection.close()


@app.context_processor
def inject_default_context():
    return {"app_name": "ML Intrusion Detection & Mail Guard"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    connection = get_db_connection()
    total_scans = connection.execute(
        "SELECT COUNT(*) AS total FROM (SELECT id FROM intrusion_predictions UNION ALL SELECT id FROM mail_predictions)"
    ).fetchone()["total"]
    normal_count = connection.execute(
        "SELECT COUNT(*) AS total FROM intrusion_predictions WHERE prediction = 'NORMAL'"
    ).fetchone()["total"]
    intrusion_count = connection.execute(
        "SELECT COUNT(*) AS total FROM intrusion_predictions WHERE prediction = 'INTRUSION'"
    ).fetchone()["total"]
    mail_count = connection.execute("SELECT COUNT(*) AS total FROM mail_predictions").fetchone()["total"]
    suspicious_mail = connection.execute(
        "SELECT COUNT(*) AS total FROM mail_predictions WHERE prediction IN ('SPAM', 'PHISHING', 'SUSPICIOUS', 'SPAM-SUSPICIOUS')"
    ).fetchone()["total"]
    recent_predictions = connection.execute(
        "SELECT 'intrusion' AS type, timestamp, prediction, confidence, risk_level, details FROM intrusion_predictions "
        "UNION ALL "
        "SELECT 'mail_guard' AS type, timestamp, prediction, confidence, risk_level, details FROM mail_predictions "
        "ORDER BY timestamp DESC LIMIT 6"
    ).fetchall()
    connection.close()
    return render_template(
        "dashboard.html",
        total_scans=total_scans,
        normal_count=normal_count,
        intrusion_count=intrusion_count,
        mail_count=mail_count,
        suspicious_mail=suspicious_mail,
        recent_predictions=recent_predictions,
    )


@app.route("/intrusion")
def intrusion_page():
    from utils.preprocessing import INTRUSION_FEATURES

    return render_template("intrusion.html", features=INTRUSION_FEATURES)


@app.route("/predict_intrusion", methods=["POST"])
def predict_intrusion():
    form_data = request.form.to_dict()
    try:
        validated_input = validate_intrusion_input(form_data)
    except ValueError as exc:
        return render_template("intrusion.html", error=str(exc), features=[
            "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Mean",
            "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "Average Packet Size", "Subflow Fwd Bytes"
        ], form_data=form_data)

    try:
        prediction, confidence, risk_level, explanation = predict_intrusion_record(validated_input)
    except FileNotFoundError:
        return render_template(
            "intrusion.html",
            error="ML model is not available. Please train the model using the training script.",
            features=[
                "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Mean",
                "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "Average Packet Size", "Subflow Fwd Bytes"
            ],
            form_data=form_data,
        )

    save_intrusion_prediction(prediction, confidence, risk_level, explanation, validated_input)

    return render_template(
        "intrusion.html",
        result={
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "explanation": explanation,
        },
        features=[
            "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
            "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Mean",
            "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "Average Packet Size", "Subflow Fwd Bytes"
        ],
        form_data=form_data,
    )


@app.route("/mail-guard")
def mail_guard_page():
    return render_template("mail_guard.html")


@app.route("/predict_mail", methods=["POST"])
def predict_mail():
    sender = (request.form.get("sender") or "").strip()
    subject = (request.form.get("subject") or "").strip()
    message = (request.form.get("message") or "").strip()
    url = (request.form.get("url") or "").strip()

    if not sender or not subject or not message:
        return render_template(
            "mail_guard.html",
            error="Please enter the sender, subject, and message before scanning.",
            form_data={"sender": sender, "subject": subject, "message": message, "url": url},
        )

    try:
        prediction, confidence, risk_level, explanation, indicators = predict_mail_text(sender, subject, message, url)
    except FileNotFoundError:
        return render_template(
            "mail_guard.html",
            error="Mail Guard model is not available. Please train the model using the training script.",
            form_data={"sender": sender, "subject": subject, "message": message, "url": url},
        )

    save_mail_prediction(sender, subject, prediction, confidence, risk_level, indicators, explanation, message)

    return render_template(
        "mail_guard.html",
        result={
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "risk_level": risk_level,
            "explanation": explanation,
            "indicators": indicators,
        },
        form_data={"sender": sender, "subject": subject, "message": message, "url": url},
    )


@app.route("/history")
def history():
    type_filter = request.args.get("type", "all").strip().lower()
    status_filter = request.args.get("status", "all").strip().lower()
    query = ""
    params = []

    query = """
        SELECT 'intrusion' AS type, id, timestamp, prediction, confidence, risk_level, details, input_data AS extra
        FROM intrusion_predictions
        UNION ALL
        SELECT 'mail_guard' AS type, id, timestamp, prediction, confidence, risk_level, details, message AS extra
        FROM mail_predictions
    """

    if type_filter != "all":
        query = f"SELECT * FROM ({query}) WHERE type = ?"
        params.append(type_filter)

    if status_filter != "all":
        query = f"SELECT * FROM ({query}) WHERE prediction LIKE ?"
        if status_filter == "normal":
            params.append("NORMAL")
        elif status_filter == "safe":
            params.append("SAFE")
        elif status_filter == "suspicious":
            params.append("%SUSPICIOUS%")
        elif status_filter == "intrusion":
            params.append("INTRUSION")
        elif status_filter == "spam":
            params.append("%SPAM%")
        else:
            params.append(f"%{status_filter.upper()}%")

    query += " ORDER BY timestamp DESC"

    connection = get_db_connection()
    rows = connection.execute(query, params).fetchall()
    connection.close()
    return render_template("history.html", rows=rows, type_filter=type_filter, status_filter=status_filter)


@app.route("/dataset")
def dataset():
    from utils.preprocessing import DATASET_INFO

    return render_template("dataset.html", dataset_info=DATASET_INFO)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/clear-history", methods=["POST"])
def clear_history():
    connection = get_db_connection()
    connection.execute("DELETE FROM intrusion_predictions")
    connection.execute("DELETE FROM mail_predictions")
    connection.commit()
    connection.close()
    return render_template("history.html", rows=[], cleared=True)


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("404.html", message="A server error occurred. Please try again."), 500


init_db()

if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
    )

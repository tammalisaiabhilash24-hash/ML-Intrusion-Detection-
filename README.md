# ML Intrusion Detection System with Mail Guard

This project is a beginner-friendly cybersecurity application built with Flask, Python, and scikit-learn. It includes:

- Network intrusion detection using a trained machine learning model
- Mail Guard analysis for suspicious or phishing-like messages
- SQLite database for storing scan history
- Dashboard with prediction statistics
- Separate training scripts for retraining the models

## Project title

ML Intrusion Detection System with Mail Guard

## Overview

The project simulates a small cybersecurity analysis tool that processes:

1. Network traffic features
2. Email subject/body content
3. Suspicious indicators such as phishing links, urgency, password requests, and financial pressure

The system is designed for local use and demonstration on localhost. It does not provide guaranteed cyber protection and should be treated as an ML-based assistance tool.

## Current project status

This workspace contains a working local Flask app and saved model artifacts. The production app loads artifacts from `models/` and does not require raw training datasets at runtime. The training scripts use public datasets and do not generate demo training data.

## Features

- Home page with cybersecurity-themed landing design
- Dashboard showing total scans, normal traffic, intrusions, mail scans, and suspicious emails
- Intrusion detection form with validated numeric inputs
- Mail Guard email scanner with suspicious indicator reporting
- Prediction history with filtering
- Dataset and model information page
- About / how-it-works page
- SQLite-backed history storage
- Training scripts for retraining both models

## Tech stack

- Flask
- Python
- SQLite
- pandas
- numpy
- scikit-learn
- joblib

## Recommended dataset sources

This project is structured to work with a real intrusion dataset such as CICIDS2017. The training pipeline expects CSV files in `data/raw/`.

Recommended intrusion dataset:
- CIC-IDS-2017 (CSV version)

Recommended mail dataset for real-world spam or phishing text classification:
- SMS Spam Collection or Enron spam corpus

If you do not have the large dataset file available, the included training scripts can generate a small demo dataset for local verification, but the project is designed to be replaced with the real datasets for proper evaluation.

## Folder structure

```text
project_root/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── database/
│   └── app.db
├── models/
│   ├── intrusion_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── mail_guard_model.pkl
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/
│   ├── 404.html
│   ├── about.html
│   ├── base.html
│   ├── dashboard.html
│   ├── dataset.html
│   ├── history.html
│   ├── index.html
│   ├── intrusion.html
│   ├── mail_guard.html
├── training/
│   ├── evaluate_models.py
│   ├── train_intrusion_model.py
│   └── train_mail_guard.py
├── utils/
│   ├── intrusion_predictor.py
│   ├── mail_guard.py
│   └── preprocessing.py
└── venv/
```

## Local setup

### 1. Create virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place datasets (optional but recommended)

Put the real CSV data in the `data/raw/` directory.

Suggested files:

- `data/raw/CICIDS2017.csv` for intrusion detection data
- `data/raw/spam.csv` or similar for mail guard training

If the dataset is not yet available, you can still train the demo model locally using the built-in training scripts.

### 4. Train the models

```bash
python training/train_intrusion_model.py
python training/train_mail_guard.py
```

### 5. Run the Flask app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Deploy to Render

The included `render.yaml` configures a Render Web Service using Gunicorn, binds to Render's `PORT`, generates a `SECRET_KEY`, and stores the SQLite database under `/tmp`. Deploy from the repository with Render's Blueprint workflow, or use these Web Service commands:

- Build: `pip install -r requirements.txt`
- Start: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 app:app`

The saved model files in `models/` are required at runtime; raw CSV files and virtual environments are not. Render's `/tmp` storage is ephemeral, so prediction history may be cleared on restart or redeploy. Use persistent database storage if durable history is required.

## How intrusion detection works

1. User enters network flow metrics such as packet counts and byte lengths.
2. The app validates the values.
3. The same feature order used at training time is applied.
4. The scaler and model process the numeric features.
5. The model predicts whether the traffic is normal or intrusive.
6. The result is saved in SQLite.

## How Mail Guard works

1. User enters the sender, subject, message body, and optional URL.
2. The app combines the text into a single message string.
3. A TF-IDF vectorizer converts the text into numeric features.
4. A machine learning classifier predicts whether the message is safe or suspicious.
5. The system also checks for common indicators such as urgency, password requests, and financial pressure.
6. The result is stored in the SQLite database.

## Model evaluation

The training scripts compute:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- class-wise metrics where applicable

These metrics are printed to the console after training.

## Database

The app uses SQLite and creates the database automatically at startup if needed.

Tables include:

- `intrusion_predictions`
- `mail_predictions`

## Common errors and solutions

### Model file not found

Run:

```bash
python training/train_intrusion_model.py
python training/train_mail_guard.py
```

### Dataset not found

Place the CSV file in `data/raw/` and rerun the training script.

### Import errors

Create the virtual environment and install the requirements again:

```bash
pip install -r requirements.txt
```

### Flask port already in use

Use a different port or stop the process already bound to 5000.

## Future improvements

- Add more intrusion features from a full CICIDS2017 preprocessing pipeline
- Collect more real spam/phishing samples for improved model quality
- Add user login and authentication
- Add a more advanced dashboard
- Add report export functionality

## Important note

This application follows a local educational and demonstration workflow. It does not claim guaranteed real-world cyber protection. The prediction system is ML-based and should be used as a learning and testing tool.

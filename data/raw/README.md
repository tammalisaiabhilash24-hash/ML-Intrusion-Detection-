# Dataset Location

This project is designed to work with real cybersecurity datasets.

## Intrusion Dataset

Place the intrusion detection CSV inside this folder.

Recommended dataset:
- CIC-IDS-2017 (CSV version)

Suggested filename:
- `CICIDS2017.csv`

The training pipeline will look for a CSV file in `data/raw/` and use the first valid one it finds.

## Mail Dataset

For mail guard training, place a spam/ham text dataset here if you want to use a real-world dataset.

Possible examples:
- `spam.csv`
- `mail_dataset.csv`

The script expects at least two columns such as:
- `text` or `message`
- `label` or `target`

If no dataset is found, the training script creates a small demo dataset so the project still runs locally for testing.

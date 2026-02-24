# Sentiment

A small, runnable sentiment-analysis application built with scikit-learn.

## What this repo contains

- `app.py` - main CLI application for training, evaluating, and predicting sentiment labels from CSV files.
- Data files you can use immediately:
  - `traindata.csv` (id, label, text)
  - `traindata1.csv` (label, short text)
  - `trainset.csv`, `mobile.csv`, `posneg.csv`, `train.csv`
- Older experimentation scripts (`sample1.py`, `svmtrain.py`, `trainclassi.py`, `test1.py`) kept for reference.

## Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

### 1) Evaluate model quality on a dataset

Using `traindata.csv` (`text` at column 2, `label` at column 1):

```bash
python app.py --data traindata.csv --text-col 2 --label-col 1
```

This runs a train/test split, prints accuracy, and a classification report.

### 2) Predict sentiment for custom text

Using `traindata1.csv` (`text` at column 1, `label` at column 0):

```bash
python app.py --data traindata1.csv --text-col 1 --label-col 0 --predict "front camera is bad"
```

### 3) Datasets with header rows

If your CSV has a header row, add `--has-header` and set columns accordingly:

```bash
python app.py --data train.csv --text-col 2 --label-col 0 --has-header
```

## CLI options

```text
--data <path>          Path to CSV file (default: traindata.csv)
--text-col <int>       Zero-based index of text column (default: 2)
--label-col <int>      Zero-based index of label column (default: 1)
--has-header           Use if first row is header
--test-size <float>    Test split ratio in evaluation mode (default: 0.2)
--random-state <int>   Random seed (default: 42)
--predict "..."        If provided, predicts one text; otherwise runs evaluation
```

## Notes

- This app uses a simple `CountVectorizer + TF-IDF + MultinomialNB` pipeline.
- Labels are treated as strings from the CSV (e.g., `0`/`1`).

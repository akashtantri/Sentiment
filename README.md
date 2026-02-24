# Sentiment

A small, runnable sentiment-analysis application built with scikit-learn.

## Project structure

```text
.
├── app.py                     # Thin entrypoint
├── src/sentiment/             # Application package
│   ├── cli.py
│   ├── data.py
│   └── model.py
├── resources/                 # CSV datasets
│   ├── traindata.csv
│   ├── traindata1.csv
│   ├── train.csv
│   ├── trainset.csv
│   ├── mobile.csv
│   ├── posneg.csv
│   └── traindata - Copy.csv
└── archive/legacy_scripts/    # Older experimentation scripts
```

## Requirements

- Python 3.9+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start

### 1) Evaluate model quality on a dataset

Using `resources/traindata.csv` (`text` at column 2, `label` at column 1):

```bash
python app.py --data resources/traindata.csv --text-col 2 --label-col 1
```

This runs a train/test split, prints accuracy, and a classification report.

### 2) Predict sentiment for custom text

Using `resources/traindata1.csv` (`text` at column 1, `label` at column 0):

```bash
python app.py --data resources/traindata1.csv --text-col 1 --label-col 0 --predict "front camera is bad"
```

### 3) Datasets with header rows

If your CSV has a header row, add `--has-header` and set columns accordingly:

```bash
python app.py --data resources/train.csv --text-col 2 --label-col 0 --has-header
```

## CLI options

```text
--data <path>          Path to CSV file (default: resources/traindata.csv)
--text-col <int>       Zero-based index of text column (default: 2)
--label-col <int>      Zero-based index of label column (default: 1)
--has-header           Use if first row is header
--test-size <float>    Test split ratio in evaluation mode (default: 0.2)
--random-state <int>   Random seed (default: 42)
--predict "..."        If provided, predicts one text; otherwise runs evaluation
```

## Notes

- The app uses a `CountVectorizer + TF-IDF + MultinomialNB` pipeline.
- Labels are treated as strings from the CSV (for example, `0`/`1`).

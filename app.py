#!/usr/bin/env python3
"""Runnable sentiment analysis application.

Example:
  python app.py --data traindata.csv --text-col 2 --label-col 1
  python app.py --data traindata1.csv --text-col 1 --label-col 0 --predict "front camera is bad"
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



def load_csv_dataset(
    data_path: Path,
    text_col: int,
    label_col: int,
    has_header: bool = False,
) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []

    with data_path.open("r", encoding="utf-8", errors="replace") as fp:
        reader = csv.reader(fp)
        if has_header:
            next(reader, None)

        for i, row in enumerate(reader, start=1):
            if not row:
                continue
            if max(text_col, label_col) >= len(row):
                raise ValueError(
                    f"Row {i} in {data_path} does not have required column index. "
                    f"Expected at least {max(text_col, label_col) + 1} columns, got {len(row)}."
                )
            text = row[text_col].strip()
            label = row[label_col].strip()
            if text:
                texts.append(text)
                labels.append(label)

    if not texts:
        raise ValueError(f"No valid text rows found in {data_path}.")

    return texts, labels



def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", CountVectorizer(ngram_range=(1, 2))),
            ("tfidf", TfidfTransformer()),
            ("classifier", MultinomialNB()),
        ]
    )



def run_evaluation(
    texts: List[str],
    labels: List[str],
    test_size: float,
    random_state: int,
) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print("=== Evaluation ===")
    print(f"Train samples: {len(x_train)}")
    print(f"Test samples:  {len(x_test)}")
    print(f"Accuracy:      {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))



def run_prediction(texts: List[str], labels: List[str], input_text: str) -> None:
    model = build_pipeline()
    model.fit(texts, labels)
    prediction = model.predict([input_text])[0]

    print("=== Prediction ===")
    print(f"Input:  {input_text}")
    print(f"Label:  {prediction}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a sentiment model from a CSV dataset.")
    parser.add_argument("--data", default="traindata.csv", help="Path to CSV dataset.")
    parser.add_argument("--text-col", type=int, default=2, help="Zero-based text column index.")
    parser.add_argument("--label-col", type=int, default=1, help="Zero-based label column index.")
    parser.add_argument("--has-header", action="store_true", help="Set when CSV has a header row.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio for evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--predict", help="Optional text to classify. If omitted, runs evaluation.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    texts, labels = load_csv_dataset(
        data_path=data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        has_header=args.has_header,
    )

    if args.predict:
        run_prediction(texts, labels, args.predict)
    else:
        run_evaluation(texts, labels, args.test_size, args.random_state)


if __name__ == "__main__":
    main()

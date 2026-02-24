"""Command-line interface for sentiment classification."""

from __future__ import annotations

import argparse
from pathlib import Path

from sentiment.data import load_csv_dataset
from sentiment.model import evaluate, predict


DEFAULT_DATASET = Path("resources/traindata.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a sentiment model from a CSV dataset.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Path to CSV dataset.")
    parser.add_argument("--text-col", type=int, default=2, help="Zero-based text column index.")
    parser.add_argument("--label-col", type=int, default=1, help="Zero-based label column index.")
    parser.add_argument("--has-header", action="store_true", help="Set when CSV has a header row.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio for evaluation.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--predict", help="Optional text to classify. If omitted, runs evaluation.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = load_csv_dataset(
        data_path=data_path,
        text_col=args.text_col,
        label_col=args.label_col,
        has_header=args.has_header,
    )

    if args.predict:
        prediction = predict(dataset, args.predict)
        print("=== Prediction ===")
        print(f"Input:  {args.predict}")
        print(f"Label:  {prediction}")
        return

    result = evaluate(dataset, args.test_size, args.random_state)
    print("=== Evaluation ===")
    print(f"Train samples: {result.train_samples}")
    print(f"Test samples:  {result.test_samples}")
    print(f"Accuracy:      {result.accuracy:.4f}")
    print("\nClassification report:")
    print(result.report)

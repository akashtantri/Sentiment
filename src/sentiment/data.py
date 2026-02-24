"""Dataset loading helpers for sentiment classification."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    texts: list[str]
    labels: list[str]


def load_csv_dataset(
    data_path: Path,
    text_col: int,
    label_col: int,
    has_header: bool = False,
) -> Dataset:
    texts: list[str] = []
    labels: list[str] = []

    with data_path.open("r", encoding="utf-8", errors="replace") as fp:
        reader = csv.reader(fp)
        if has_header:
            next(reader, None)

        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if max(text_col, label_col) >= len(row):
                raise ValueError(
                    f"Row {row_number} in {data_path} does not have required column index. "
                    f"Expected at least {max(text_col, label_col) + 1} columns, got {len(row)}."
                )

            text = row[text_col].strip()
            label = row[label_col].strip()
            if text:
                texts.append(text)
                labels.append(label)

    if not texts:
        raise ValueError(f"No valid text rows found in {data_path}.")

    return Dataset(texts=texts, labels=labels)

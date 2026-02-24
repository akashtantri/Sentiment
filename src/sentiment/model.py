"""Model construction and execution logic."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sentiment.data import Dataset


@dataclass(frozen=True)
class EvaluationResult:
    train_samples: int
    test_samples: int
    accuracy: float
    report: str


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", CountVectorizer(ngram_range=(1, 2))),
            ("tfidf", TfidfTransformer()),
            ("classifier", MultinomialNB()),
        ]
    )


def evaluate(dataset: Dataset, test_size: float, random_state: int) -> EvaluationResult:
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels if len(set(dataset.labels)) > 1 else None,
    )

    model = build_pipeline()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return EvaluationResult(
        train_samples=len(x_train),
        test_samples=len(x_test),
        accuracy=accuracy_score(y_test, predictions),
        report=classification_report(y_test, predictions, zero_division=0),
    )


def predict(dataset: Dataset, input_text: str) -> str:
    model = build_pipeline()
    model.fit(dataset.texts, dataset.labels)
    return str(model.predict([input_text])[0])

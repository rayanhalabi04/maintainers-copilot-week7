import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

TRAIN_PATH = Path("data/processed/train.jsonl")
VAL_PATH = Path("data/processed/val.jsonl")
TEST_PATH = Path("data/processed/test.jsonl")

ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")

LABELS = ["bug", "feature", "docs", "question"]


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def evaluate(model: Pipeline, df: pd.DataFrame, split_name: str) -> dict:
    x = df["text"]
    y = df["label"]

    start = time.time()
    preds = model.predict(x)
    prediction_time = time.time() - start

    return {
        "split": split_name,
        "accuracy": accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "avg_prediction_latency_ms": (prediction_time / len(x)) * 1000,
        "labels": LABELS,
        "classification_report": classification_report(
            y,
            preds,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, preds, labels=LABELS).tolist(),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_jsonl(TRAIN_PATH)
    val_df = load_jsonl(VAL_PATH)
    test_df = load_jsonl(TEST_PATH)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("Training classical model...")
    start_train = time.time()
    model.fit(train_df["text"], train_df["label"])
    train_time = time.time() - start_train

    val_metrics = evaluate(model, val_df, "validation")
    test_metrics = evaluate(model, test_df, "test")

    full_report = {
        "model": "tfidf_logistic_regression",
        "train_time_seconds": train_time,
        "validation": val_metrics,
        "test": test_metrics,
    }

    joblib.dump(model, ARTIFACT_DIR / "classical_classifier.joblib")

    with (REPORT_DIR / "classical_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\nDone.")
    print("Validation accuracy:", val_metrics["accuracy"])
    print("Validation macro-F1:", val_metrics["macro_f1"])
    print("Test accuracy:", test_metrics["accuracy"])
    print("Test macro-F1:", test_metrics["macro_f1"])
    print("Saved model: artifacts/classical_classifier.joblib")
    print("Saved report: reports/classical_metrics.json")


if __name__ == "__main__":
    main()

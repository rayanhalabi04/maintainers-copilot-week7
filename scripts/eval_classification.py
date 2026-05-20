#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.domain.classification import ClassifyRequest  # noqa: E402
from app.services.classifier_service import ClassifierService  # noqa: E402


GOLDEN_PATH = ROOT / "data" / "golden" / "classification_golden.jsonl"
EVAL_REPORT_PATH = ROOT / "eval_report.json"
PREDICTIONS_PATH = ROOT / "data" / "evals" / "classification_predictions.jsonl"
CONFUSION_MATRIX_PATH = ROOT / "data" / "evals" / "classification_confusion_matrix.json"
THRESHOLDS_PATH = ROOT / "eval_thresholds.yaml"
LABELS = ["bug", "feature", "docs", "question"]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        fail(f"Missing golden file: {path.relative_to(ROOT)}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"Invalid JSON on line {line_number}: {exc}")
        rows.append(row)
    if not rows:
        fail(f"No examples found in {path.relative_to(ROOT)}")
    return rows


def input_text(example: dict[str, Any]) -> str:
    text = str(example.get("text") or "").strip()
    if text:
        return text
    title = str(example.get("title") or "").strip()
    body = str(example.get("body") or "").strip()
    return f"{title}\n\n{body}".strip()


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for item in predictions if item["correct"])
    per_class = {}
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    label_index = {label: index for index, label in enumerate(LABELS)}

    for item in predictions:
        expected = item["expected_label"]
        predicted = item["predicted_label"]
        if expected in label_index and predicted in label_index:
            matrix[label_index[expected]][label_index[predicted]] += 1

    for label in LABELS:
        true_positive = sum(
            1
            for item in predictions
            if item["expected_label"] == label and item["predicted_label"] == label
        )
        false_positive = sum(
            1
            for item in predictions
            if item["expected_label"] != label and item["predicted_label"] == label
        )
        false_negative = sum(
            1
            for item in predictions
            if item["expected_label"] == label and item["predicted_label"] != label
        )
        support = sum(1 for item in predictions if item["expected_label"] == label)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1(precision, recall),
            "support": support,
        }

    macro_f1 = sum(values["f1"] for values in per_class.values()) / len(LABELS)
    return {
        "examples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABELS,
            "rows": "expected",
            "columns": "predicted",
            "matrix": matrix,
        },
    }


def load_thresholds(path: Path) -> dict[str, float]:
    thresholds = {
        "classification_accuracy_min": 0.5,
        "classification_macro_f1_min": 0.5,
    }
    if not path.exists():
        return thresholds
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(" ") or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in thresholds and value:
            try:
                thresholds[key] = float(value)
            except ValueError:
                fail(f"Invalid numeric threshold for {key}: {value!r}")
    return thresholds


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    examples = load_jsonl(GOLDEN_PATH)
    service = ClassifierService()
    predictions = []

    for example in examples:
        text = input_text(example)
        if not text:
            fail(f"Golden example {example.get('id') or example.get('issue_number')} has no input text")
        expected_label = str(example.get("expected_label") or "")
        response = service.classify(ClassifyRequest(title=text, body=""))
        predictions.append(
            {
                "id": example.get("id"),
                "issue_number": example.get("issue_number"),
                "source_split": example.get("source_split"),
                "expected_label": expected_label,
                "predicted_label": response.label,
                "confidence": response.confidence,
                "probabilities": response.probabilities,
                "correct": response.label == expected_label,
            }
        )

    metrics = compute_metrics(predictions)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    metrics["thresholds"] = thresholds

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in predictions),
        encoding="utf-8",
    )
    write_json(CONFUSION_MATRIX_PATH, metrics["confusion_matrix"])

    report = {}
    if EVAL_REPORT_PATH.exists():
        try:
            report = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"{EVAL_REPORT_PATH.relative_to(ROOT)} is not valid JSON: {exc}")
    report["classification"] = metrics
    write_json(EVAL_REPORT_PATH, report)

    print(json.dumps({"classification": metrics}, indent=2))

    failures = []
    if metrics["accuracy"] < thresholds["classification_accuracy_min"]:
        failures.append(
            f"accuracy {metrics['accuracy']:.4f} < {thresholds['classification_accuracy_min']:.4f}"
        )
    if metrics["macro_f1"] < thresholds["classification_macro_f1_min"]:
        failures.append(
            f"macro_f1 {metrics['macro_f1']:.4f} < {thresholds['classification_macro_f1_min']:.4f}"
        )
    if failures:
        fail("Classification eval failed thresholds: " + "; ".join(failures))


if __name__ == "__main__":
    main()

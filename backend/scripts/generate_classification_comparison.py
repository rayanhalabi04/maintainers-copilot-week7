import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
OUTPUT_PATH = REPORTS_DIR / "classification_comparison.json"


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def per_class_f1(split_metrics: dict[str, Any]) -> dict[str, float] | None:
    if "per_class_f1" in split_metrics:
        return split_metrics["per_class_f1"]

    report = split_metrics.get("classification_report", {})
    labels = split_metrics.get("labels", ["bug", "feature", "docs", "question"])
    if not report:
        return None
    return {
        label: float(report[label]["f1-score"])
        for label in labels
        if label in report
    }


def summarize_model(
    name: str,
    metrics: dict[str, Any] | None,
    report_path: Path,
) -> dict[str, Any]:
    if metrics is None:
        return {
            "status": "missing",
            "report_path": str(report_path),
            "notes": "Metrics file was not found.",
        }

    if metrics.get("status") == "skipped":
        return {
            "status": "skipped",
            "model": metrics.get("model"),
            "report_path": str(report_path),
            "reason": metrics.get("reason"),
            "notes": metrics.get("notes", []),
        }

    test = metrics.get("test") or {}
    validation = metrics.get("validation")
    latency = test.get("avg_prediction_latency_ms")
    if latency is None:
        latency = test.get("avg_latency_ms")

    return {
        "status": metrics.get("status", "completed"),
        "model": metrics.get("model"),
        "report_path": str(report_path),
        "artifact_dir": metrics.get("artifact_dir"),
        "validation": {
            "accuracy": validation.get("accuracy"),
            "macro_f1": validation.get("macro_f1"),
            "per_class_f1": per_class_f1(validation),
            "confusion_matrix": validation.get("confusion_matrix"),
        }
        if validation
        else None,
        "test": {
            "accuracy": test.get("accuracy"),
            "macro_f1": test.get("macro_f1"),
            "per_class_f1": per_class_f1(test),
            "confusion_matrix": test.get("confusion_matrix"),
            "avg_prediction_latency_ms": latency,
        },
        "train_time_seconds": metrics.get("train_time_seconds"),
        "cost": metrics.get("cost"),
        "confusion_matrix_path": metrics.get("confusion_matrix_path"),
        "notes": metrics.get("notes", []),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "classical": REPORTS_DIR / "classical_metrics.json",
        "transformer": REPORTS_DIR / "transformer_metrics.json",
        "llm_baseline": REPORTS_DIR / "llm_baseline_metrics.json",
    }
    comparison = {
        "task": "four_label_github_issue_classification",
        "labels": ["bug", "feature", "docs", "question"],
        "splits": {
            "train": "data/processed/train.jsonl",
            "validation": "data/processed/val.jsonl",
            "test": "data/processed/test.jsonl",
        },
        "models": {
            name: summarize_model(name, read_json(path), path)
            for name, path in paths.items()
        },
        "notes": [
            "All completed models use the same processed train/validation/test files.",
            "Cost is only present for the LLM baseline when token pricing environment variables are supplied.",
        ],
    }

    OUTPUT_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Saved comparison report: {OUTPUT_PATH}")

    for name, summary in comparison["models"].items():
        test = summary.get("test") or {}
        print(
            name,
            "status=", summary["status"],
            "test_macro_f1=", test.get("macro_f1"),
            "test_accuracy=", test.get("accuracy"),
        )


if __name__ == "__main__":
    main()

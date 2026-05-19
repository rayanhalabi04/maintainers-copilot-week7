import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "train.jsonl"
VAL_PATH = ROOT / "data" / "processed" / "val.jsonl"
TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"

ARTIFACT_DIR = ROOT / "artifacts" / "transformer_classifier"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "transformer_metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "transformer_confusion_matrix.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
LABELS = ["bug", "feature", "docs", "question"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dataset_hash(paths: list[Path]) -> str:
    sha = hashlib.sha256()
    for path in paths:
        sha.update(path.name.encode("utf-8"))
        sha.update(path.read_bytes())
    return sha.hexdigest()


class IssueDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        label_to_id: dict[str, int],
        max_length: int,
        max_samples: int | None = None,
    ) -> None:
        if max_samples is not None:
            rows = rows[:max_samples]

        self.texts = [str(row["text"]) for row in rows]
        self.labels = [label_to_id[str(row["label"])] for row in rows]
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def validate_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    if not rows:
        raise ValueError(f"{split_name} split is empty.")

    missing = [key for key in ["text", "label"] if key not in rows[0]]
    if missing:
        raise ValueError(f"{split_name} split is missing required keys: {missing}")

    unknown = sorted({str(row["label"]) for row in rows} - set(LABELS))
    if unknown:
        raise ValueError(f"{split_name} split has unsupported labels: {unknown}")


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def evaluate_split(
    trainer: Trainer,
    dataset: Dataset,
    split_name: str,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    start = time.perf_counter()
    output = trainer.predict(dataset)
    elapsed = time.perf_counter() - start

    y_true = output.label_ids
    y_pred = np.argmax(output.predictions, axis=-1)
    labels = list(range(len(id_to_label)))
    label_names = [id_to_label[i] for i in labels]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    avg_latency_ms = float((elapsed / len(dataset)) * 1000)
    per_class_f1 = {
        label: float(report[label]["f1-score"])
        for label in label_names
    }

    print(f"\n{split_name} accuracy: {accuracy:.4f}")
    print(f"{split_name} macro-F1: {macro_f1:.4f}")
    print(f"{split_name} average prediction latency: {avg_latency_ms:.2f} ms/example")
    print(f"{split_name} per-class F1:")
    for label, score in per_class_f1.items():
        print(f"  {label}: {score:.4f}")
    print(f"{split_name} confusion matrix labels: {label_names}")
    print(matrix)

    return {
        "split": split_name.lower(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_prediction_latency_ms": avg_latency_ms,
        "labels": label_names,
        "per_class_f1": per_class_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer issue classifier.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full training split.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for full validation and test splits.",
    )
    return parser


def write_model_card(
    args: argparse.Namespace,
    data_hash: str,
    train_size: int,
    val_size: int,
    test_size: int,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    model_card = f"""# Transformer Issue Classifier Model Card

## Model

- Base model: `{args.model_name}`
- Task: GitHub issue classification
- Labels: `{", ".join(LABELS)}`
- Architecture: encoder transformer with a sequence classification head

## Data

- Train split: `{TRAIN_PATH}`
- Validation split: `{VAL_PATH}`
- Test split: `{TEST_PATH}`
- Split sizes: train={train_size}, validation={val_size}, test={test_size}
- Dataset SHA-256: `{data_hash}`

## Hyperparameters

- Epochs: {args.epochs}
- Learning rate: {args.learning_rate}
- Batch size: {args.batch_size}
- Max sequence length: {args.max_length}
- Weight decay: {args.weight_decay}
- Seed: {args.seed}
- Max train samples: {args.max_train_samples}
- Max eval samples: {args.max_eval_samples}

## Results

### Validation

- Accuracy: {val_metrics["accuracy"]}
- Macro-F1: {val_metrics["macro_f1"]}
- Per-class F1: `{json.dumps(val_metrics["per_class_f1"], sort_keys=True)}`

### Test

- Accuracy: {test_metrics["accuracy"]}
- Macro-F1: {test_metrics["macro_f1"]}
- Per-class F1: `{json.dumps(test_metrics["per_class_f1"], sort_keys=True)}`
- Average prediction latency: {test_metrics["avg_prediction_latency_ms"]} ms/example

## Notes

This model is trained and evaluated on the same processed splits as the classical ML baseline and the LLM baseline.
"""
    MODEL_CARD_PATH.write_text(model_card, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    print("Training fine-tuned transformer classifier...")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)
    test_rows = load_jsonl(TEST_PATH)

    validate_rows(train_rows, "train")
    validate_rows(val_rows, "validation")
    validate_rows(test_rows, "test")

    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    print(f"Labels: {LABELS}")
    print(f"Train/validation/test sizes: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}")
    print(f"Base model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = IssueDataset(
        train_rows,
        tokenizer,
        label_to_id,
        args.max_length,
        args.max_train_samples,
    )
    val_dataset = IssueDataset(
        val_rows,
        tokenizer,
        label_to_id,
        args.max_length,
        args.max_eval_samples,
    )
    test_dataset = IssueDataset(
        test_rows,
        tokenizer,
        label_to_id,
        args.max_length,
        args.max_eval_samples,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(ARTIFACT_DIR / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(ARTIFACT_DIR / "logs"),
        logging_steps=20,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    start_train = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start_train

    print("\nEvaluating best transformer model...")
    val_metrics = evaluate_split(trainer, val_dataset, "Validation", id_to_label)
    test_metrics = evaluate_split(trainer, test_dataset, "Test", id_to_label)

    trainer.save_model(str(ARTIFACT_DIR))
    tokenizer.save_pretrained(str(ARTIFACT_DIR))

    data_hash = dataset_hash([TRAIN_PATH, VAL_PATH, TEST_PATH])
    hyperparameters = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
    }
    metrics = {
        "model": args.model_name,
        "task": "issue_classification",
        "labels": LABELS,
        "label_to_id": label_to_id,
        "data_hash": data_hash,
        "train_time_seconds": float(train_time),
        "hyperparameters": hyperparameters,
        "validation": val_metrics,
        "test": test_metrics,
        "artifact_dir": str(ARTIFACT_DIR),
        "confusion_matrix_path": str(CONFUSION_MATRIX_PATH),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CONFUSION_MATRIX_PATH.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "validation": val_metrics["confusion_matrix"],
                "test": test_metrics["confusion_matrix"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_model_card(
        args,
        data_hash,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
        val_metrics,
        test_metrics,
    )

    print("\nDone.")
    print(f"Train time seconds: {train_time:.2f}")
    print(f"Saved transformer model to: {ARTIFACT_DIR}")
    print(f"Saved metrics to: {METRICS_PATH}")
    print(f"Saved confusion matrix to: {CONFUSION_MATRIX_PATH}")
    print(f"Saved model card to: {MODEL_CARD_PATH}")


if __name__ == "__main__":
    main()

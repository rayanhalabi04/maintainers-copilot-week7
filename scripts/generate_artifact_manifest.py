#!/usr/bin/env python3
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "backend" / "model_server" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "tfidf_logreg_baseline.joblib"
PREFERRED_TRAIN_PATH = ARTIFACT_DIR / "train.csv"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"
METRICS_PATH = ARTIFACT_DIR / "tfidf_logreg_metrics.json"
MANIFEST_PATH = ARTIFACT_DIR / "artifact_manifest.json"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def find_training_data() -> Path:
    if PREFERRED_TRAIN_PATH.exists():
        return PREFERRED_TRAIN_PATH

    candidates = sorted(
        path
        for path in ROOT.rglob("train.csv")
        if ".venv" not in path.parts and "site-packages" not in path.parts
    )
    if not candidates:
        fail("Could not find classifier training data file train.csv")
    return candidates[0]


def load_model_card() -> dict[str, Any]:
    if not MODEL_CARD_PATH.exists():
        fail(f"Model card not found at {relative(MODEL_CARD_PATH)}")
    try:
        return json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Model card is not valid JSON: {exc}")


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{relative(path)} is not valid JSON: {exc}")
    return data if isinstance(data, dict) else {}


def main() -> None:
    if not MODEL_PATH.exists():
        fail(f"Model artifact not found at {relative(MODEL_PATH)}")

    training_data_path = find_training_data()
    if not training_data_path.exists():
        fail(f"Training data file not found at {relative(training_data_path)}")

    model_card = load_model_card()
    metrics = load_optional_json(METRICS_PATH)
    model_sha256 = sha256_file(MODEL_PATH)
    training_data_sha256 = sha256_file(training_data_path)

    label_names = model_card.get("labels") or model_card.get("label_names") or []
    source_files_used = [
        relative(MODEL_PATH),
        relative(training_data_path),
        relative(MODEL_CARD_PATH),
    ]
    if METRICS_PATH.exists():
        source_files_used.append(relative(METRICS_PATH))

    manifest = {
        "model_name": model_card.get("deployment_model", "tfidf_logreg_baseline"),
        "model_type": model_card.get("model_type", "TF-IDF + Logistic Regression"),
        "label_names": label_names,
        "model_artifact_path": relative(MODEL_PATH),
        "model_sha256": model_sha256,
        "training_data_path": relative(training_data_path),
        "training_data_sha256": training_data_sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo": metrics.get("repo") or model_card.get("repo") or model_card.get("dataset"),
        "source_files_used": source_files_used,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    model_card["training_data_sha256"] = training_data_sha256
    model_card["model_sha256"] = model_sha256
    model_card["artifact_manifest_path"] = relative(MANIFEST_PATH)
    MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {relative(MANIFEST_PATH)}")
    print(f"Updated {relative(MODEL_CARD_PATH)}")


if __name__ == "__main__":
    main()

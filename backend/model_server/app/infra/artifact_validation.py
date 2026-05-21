import hashlib
import json
from pathlib import Path
from typing import Any


class ArtifactValidationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_classifier_artifacts(
    artifact_dir: Path,
    model_filename: str = "tfidf_logreg_baseline.joblib",
) -> dict[str, Any]:
    model_path = artifact_dir / model_filename
    model_card_path = artifact_dir / "model_card.json"
    manifest_path = artifact_dir / "artifact_manifest.json"

    if not model_path.exists():
        raise ArtifactValidationError(f"Missing classifier artifact: {model_path}")
    if not model_card_path.exists():
        raise ArtifactValidationError(f"Missing classifier model card: {model_card_path}")
    if not manifest_path.exists():
        raise ArtifactValidationError(f"Missing classifier artifact manifest: {manifest_path}")

    model_card = _load_json(model_card_path, "model card")
    manifest = _load_json(manifest_path, "artifact manifest")

    labels = model_card.get("label_mapping") or model_card.get("labels") or manifest.get("label_names")
    if not labels:
        raise ArtifactValidationError("Classifier label mapping/labels are missing.")

    expected_sha = manifest.get("model_sha256") or model_card.get("model_sha256")
    if not expected_sha:
        raise ArtifactValidationError("Classifier model SHA-256 is missing from manifest/model card.")

    actual_sha = sha256_file(model_path)
    if actual_sha != expected_sha:
        raise ArtifactValidationError(
            "Classifier artifact SHA-256 mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )

    return {
        "model_path": str(model_path),
        "model_card": model_card,
        "manifest": manifest,
        "model_sha256": actual_sha,
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"Invalid {label} JSON at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactValidationError(f"Invalid {label} JSON at {path}: expected object")
    return data

import json
from pathlib import Path

import pytest

from app.infra.artifact_validation import ArtifactValidationError, validate_classifier_artifacts
from app.infra.eval_thresholds import ThresholdValidationError, load_required_thresholds
from app.services import startup_validation


def write_thresholds(path: Path, rag_hit: str = "0.60") -> None:
    path.write_text(
        "rag:\n"
        f"  hit_at_5: {rag_hit}\n"
        "  mrr_at_10: 0.50\n"
        "  faithfulness: 0.55\n"
        "  answer_relevancy: 0.20\n"
        "classification_accuracy_min: 0.5\n"
        "classification_macro_f1_min: 0.5\n",
        encoding="utf-8",
    )


def test_startup_threshold_validation_rejects_missing_file(tmp_path: Path):
    with pytest.raises(ThresholdValidationError, match="Missing eval thresholds file"):
        load_required_thresholds(tmp_path / "missing.yaml")


def test_startup_threshold_validation_rejects_zero_threshold(tmp_path: Path):
    path = tmp_path / "eval_thresholds.yaml"
    write_thresholds(path, rag_hit="0")

    with pytest.raises(ThresholdValidationError, match="rag.hit_at_5"):
        load_required_thresholds(path)


def test_classifier_artifact_validation_rejects_sha_mismatch(tmp_path: Path):
    model_path = tmp_path / "tfidf_logreg_baseline.joblib"
    model_path.write_bytes(b"model-bytes")
    (tmp_path / "model_card.json").write_text(
        json.dumps({"labels": ["bug"], "model_sha256": "bad-sha"}),
        encoding="utf-8",
    )
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps({"label_names": ["bug"], "model_sha256": "bad-sha"}),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        validate_classifier_artifacts(tmp_path)


def test_strict_startup_requires_explicit_local_dev(monkeypatch):
    monkeypatch.setenv("STRICT_STARTUP_VALIDATION", "true")
    monkeypatch.delenv("LOCAL_DEV_MODE", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)

    assert startup_validation.strict_startup_enabled() is True
    assert startup_validation._local_dev_mode_enabled(strict=True) is False

    monkeypatch.setenv("LOCAL_DEV_MODE", "true")
    assert startup_validation._local_dev_mode_enabled(strict=True) is True

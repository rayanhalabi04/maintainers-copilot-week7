from pathlib import Path
import json
import joblib

from app.infra.artifact_validation import validate_classifier_artifacts


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "tfidf_logreg_baseline.joblib"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"


def load_model():
    validate_classifier_artifacts(ARTIFACT_DIR)
    return joblib.load(MODEL_PATH)


def load_model_card() -> dict:
    validate_classifier_artifacts(ARTIFACT_DIR)
    with open(MODEL_CARD_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

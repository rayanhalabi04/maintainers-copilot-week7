from pathlib import Path
import json
import joblib


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "tfidf_logreg_baseline.joblib"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def load_model_card() -> dict:
    if not MODEL_CARD_PATH.exists():
        return {
            "model_name": "tfidf_logreg_baseline",
            "warning": "model_card.json not found",
        }

    with open(MODEL_CARD_PATH, "r", encoding="utf-8") as file:
        return json.load(file)
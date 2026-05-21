from pathlib import Path


REQUIRED_RAG_THRESHOLDS = ("hit_at_5", "mrr_at_10", "faithfulness", "answer_relevancy")
REQUIRED_CLASSIFICATION_THRESHOLDS = (
    "classification_accuracy_min",
    "classification_macro_f1_min",
)


class ThresholdValidationError(RuntimeError):
    pass


def load_required_thresholds(path: Path) -> dict[str, dict[str, float] | float]:
    if not path.exists():
        raise ThresholdValidationError(f"Missing eval thresholds file: {path}")

    parsed = _parse_simple_threshold_yaml(path)
    rag = parsed.get("rag")
    if not isinstance(rag, dict):
        raise ThresholdValidationError("Missing required rag threshold section.")

    for key in REQUIRED_RAG_THRESHOLDS:
        _require_positive_threshold(rag, key, section="rag")
    for key in REQUIRED_CLASSIFICATION_THRESHOLDS:
        _require_positive_threshold(parsed, key)

    return parsed


def load_rag_thresholds(path: Path) -> dict[str, float]:
    parsed = load_required_thresholds(path)
    rag = parsed["rag"]
    if not isinstance(rag, dict):
        raise ThresholdValidationError("Missing required rag threshold section.")
    return {key: float(rag[key]) for key in REQUIRED_RAG_THRESHOLDS}


def load_classification_thresholds(path: Path) -> dict[str, float]:
    parsed = load_required_thresholds(path)
    return {key: float(parsed[key]) for key in REQUIRED_CLASSIFICATION_THRESHOLDS}


def _parse_simple_threshold_yaml(path: Path) -> dict[str, dict[str, float | None | bool] | float | None | bool]:
    parsed: dict[str, dict[str, float | None | bool] | float | None | bool] = {}
    current_section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if not raw_line.startswith(" ") and stripped.endswith(":"):
            current_section = stripped[:-1]
            parsed[current_section] = {}
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        value = _parse_threshold_value(raw_value.strip())
        if raw_line.startswith(" ") and current_section:
            section = parsed.setdefault(current_section, {})
            if isinstance(section, dict):
                section[key.strip()] = value
        else:
            current_section = None
            parsed[key.strip()] = value
    return parsed


def _parse_threshold_value(value: str) -> float | None | bool:
    lowered = value.lower()
    if lowered in {"", "null", "none", "~"}:
        return None
    if lowered in {"false", "disabled", "off"}:
        return False
    if lowered in {"true", "enabled", "on"}:
        return True
    try:
        return float(value)
    except ValueError as exc:
        raise ThresholdValidationError(f"Invalid threshold value: {value!r}") from exc


def _require_positive_threshold(
    values: dict[str, float | None | bool] | dict[str, dict[str, float | None | bool] | float | None | bool],
    key: str,
    section: str | None = None,
) -> None:
    if key not in values:
        prefix = f"{section}." if section else ""
        raise ThresholdValidationError(f"Missing required threshold: {prefix}{key}")
    value = values[key]
    if isinstance(value, bool) or value is None:
        prefix = f"{section}." if section else ""
        raise ThresholdValidationError(f"Threshold must be numeric and enabled: {prefix}{key}")
    if float(value) <= 0:
        prefix = f"{section}." if section else ""
        raise ThresholdValidationError(f"Threshold must be > 0: {prefix}{key}")

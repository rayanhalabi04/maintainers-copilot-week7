import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "processed" / "test.jsonl"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "llm_baseline_metrics.json"

LABELS = ["bug", "feature", "docs", "question"]
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_API_BASE = "https://api.openai.com/v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def estimate_cost(usage: dict[str, int]) -> dict[str, Any]:
    input_rate = os.getenv("LLM_INPUT_COST_PER_1M")
    output_rate = os.getenv("LLM_OUTPUT_COST_PER_1M")
    if not input_rate or not output_rate:
        return {
            "estimated_cost_usd": None,
            "note": "Set LLM_INPUT_COST_PER_1M and LLM_OUTPUT_COST_PER_1M to estimate cost.",
        }

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    cost = (
        (prompt_tokens / 1_000_000) * float(input_rate)
        + (completion_tokens / 1_000_000) * float(output_rate)
    )
    return {
        "estimated_cost_usd": cost,
        "input_cost_per_1m_tokens": float(input_rate),
        "output_cost_per_1m_tokens": float(output_rate),
    }


def write_skipped_report(reason: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": os.getenv("LLM_MODEL", DEFAULT_MODEL),
        "task": "issue_classification",
        "status": "skipped",
        "reason": reason,
        "labels": LABELS,
        "test": None,
        "notes": [
            "LLM baseline was not run.",
            "Set OPENAI_API_KEY to enable OpenAI-compatible chat completion evaluation.",
            "Optional: set LLM_MODEL, LLM_API_BASE, LLM_BASELINE_LIMIT, LLM_INPUT_COST_PER_1M, and LLM_OUTPUT_COST_PER_1M.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(reason)
    print(f"Saved skipped report: {METRICS_PATH}")


def parse_label(content: str) -> str:
    normalized = content.strip().lower()
    try:
        data = json.loads(normalized)
        normalized = str(data.get("label", "")).strip().lower()
    except json.JSONDecodeError:
        match = re.search(r"\b(bug|feature|docs|question)\b", normalized)
        normalized = match.group(1) if match else normalized

    if normalized not in LABELS:
        raise ValueError(f"LLM returned unsupported label: {content!r}")
    return normalized


def classify_issue(
    row: dict[str, Any],
    api_key: str,
    api_base: str,
    model: str,
) -> tuple[str, float, dict[str, int]]:
    prompt = (
        "Classify this GitHub issue into exactly one label: bug, feature, docs, question.\n"
        "Return only JSON like {\"label\":\"bug\"}.\n\n"
        f"Issue text:\n{row['text']}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "You are a precise GitHub issue triage classifier.",
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    usage = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return parse_label(content), elapsed, usage


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        write_skipped_report("OPENAI_API_KEY is not set, so the LLM baseline was skipped.")
        return

    model = os.getenv("LLM_MODEL", DEFAULT_MODEL)
    api_base = os.getenv("LLM_API_BASE", DEFAULT_API_BASE)
    limit = os.getenv("LLM_BASELINE_LIMIT")

    rows = load_jsonl(TEST_PATH)
    if limit:
        rows = rows[: int(limit)]

    y_true: list[str] = []
    y_pred: list[str] = []
    latencies: list[float] = []
    errors: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for row in tqdm(rows, desc="Evaluating LLM baseline"):
        y_true.append(str(row["label"]))
        try:
            pred, latency, row_usage = classify_issue(row, api_key, api_base, model)
            y_pred.append(pred)
            latencies.append(latency)
            for key in usage:
                usage[key] += row_usage.get(key, 0)
        except Exception as exc:
            y_pred.append("question")
            errors.append({"id": row.get("id"), "error": str(exc)})

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = {label: float(report[label]["f1-score"]) for label in LABELS}
    latency_total = sum(latencies)
    metrics = {
        "model": model,
        "task": "issue_classification",
        "status": "completed" if not errors else "completed_with_errors",
        "labels": LABELS,
        "test": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "avg_prediction_latency_ms": float((latency_total / max(len(latencies), 1)) * 1000),
            "labels": LABELS,
            "per_class_f1": per_class_f1,
            "classification_report": report,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        },
        "usage": usage,
        "cost": estimate_cost(usage),
        "num_examples": len(rows),
        "num_errors": len(errors),
        "errors": errors[:20],
        "notes": [
            "Evaluated on data/processed/test.jsonl, the same test split as the other classifiers.",
            "Rows that failed API classification are counted as question so metrics remain well-defined.",
        ],
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("LLM baseline done.")
    print("Test accuracy:", metrics["test"]["accuracy"])
    print("Test macro-F1:", metrics["test"]["macro_f1"])
    print("Average latency ms/example:", metrics["test"]["avg_prediction_latency_ms"])
    print("Estimated cost USD:", metrics["cost"]["estimated_cost_usd"])
    print(f"Saved report: {METRICS_PATH}")


if __name__ == "__main__":
    main()

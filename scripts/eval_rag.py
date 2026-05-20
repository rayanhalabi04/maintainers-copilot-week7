import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.services.rag_service import RagService  # noqa: E402


EVAL_REPORT_PATH = ROOT / "eval_report.json"
THRESHOLDS_PATH = ROOT / "eval_thresholds.yaml"


def _load_examples(golden_path: Path) -> list[dict]:
    examples = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    placeholder_rows = []
    for index, example in enumerate(examples, start=1):
        chunk_ids = example.get("ground_truth_chunk_ids") or []
        if any(str(chunk_id).startswith("placeholder") for chunk_id in chunk_ids):
            placeholder_rows.append(index)
    if placeholder_rows:
        rows = ", ".join(str(row) for row in placeholder_rows)
        raise SystemExit(
            "RAG golden contains placeholder chunk IDs. "
            f"Fix data/evals/rag_golden.jsonl before evaluation. Rows: {rows}"
        )
    return examples


def _score(
    examples: list[dict],
    retrieve: Callable[[str], list],
) -> dict[str, float | int]:
    hit5 = 0
    reciprocal_sum = 0.0

    for example in examples:
        chunks = retrieve(example["question"])
        retrieved_ids = [chunk.chunk_id for chunk in chunks]
        truth = set(example.get("ground_truth_chunk_ids", []))
        if truth & set(retrieved_ids[:5]):
            hit5 += 1
        for rank, chunk_id in enumerate(retrieved_ids[:10], start=1):
            if chunk_id in truth:
                reciprocal_sum += 1 / rank
                break

    return {
        "examples": len(examples),
        "hit_at_5": hit5 / len(examples) if examples else 0.0,
        "mrr_at_10": reciprocal_sum / len(examples) if examples else 0.0,
    }


def _load_thresholds(path: Path) -> dict[str, float]:
    thresholds = {
        "hit_at_5": 0.10,
        "mrr_at_10": 0.05,
    }
    if not path.exists():
        return thresholds

    in_rag_section = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            in_rag_section = stripped[:-1] == "rag"
            continue
        if not in_rag_section or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in thresholds and value:
            try:
                thresholds[key] = float(value)
            except ValueError as exc:
                raise SystemExit(f"Invalid numeric RAG threshold for {key}: {value!r}") from exc
    return thresholds


def _load_existing_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path.relative_to(ROOT)} is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _enforce_thresholds(metrics: dict, thresholds: dict[str, float]) -> None:
    advanced = metrics["advanced"]
    failures = []
    if advanced["hit_at_5"] < thresholds["hit_at_5"]:
        failures.append(f"advanced hit_at_5 {advanced['hit_at_5']:.4f} < {thresholds['hit_at_5']:.4f}")
    if advanced["mrr_at_10"] < thresholds["mrr_at_10"]:
        failures.append(f"advanced mrr_at_10 {advanced['mrr_at_10']:.4f} < {thresholds['mrr_at_10']:.4f}")
    if failures:
        raise SystemExit("RAG eval failed thresholds: " + "; ".join(failures))


def main() -> None:
    golden_path = ROOT / "data" / "evals" / "rag_golden.jsonl"
    examples = _load_examples(golden_path)
    service = RagService()

    thresholds = _load_thresholds(THRESHOLDS_PATH)
    rag_report = {
        "naive": _score(
            examples,
            lambda question: service.retrieve_naive(question, top_k=10),
        ),
        "advanced": _score(
            examples,
            lambda question: service.retrieve_advanced(question, top_k=10)[0],
        ),
        "thresholds": thresholds,
    }
    report = _load_existing_report(EVAL_REPORT_PATH)
    report["rag"] = rag_report
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rag": rag_report}, indent=2))
    _enforce_thresholds(rag_report, thresholds)


if __name__ == "__main__":
    main()

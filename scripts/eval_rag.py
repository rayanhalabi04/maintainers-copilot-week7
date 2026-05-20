import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.services.rag_service import RagService  # noqa: E402


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


def main() -> None:
    golden_path = ROOT / "data" / "evals" / "rag_golden.jsonl"
    examples = _load_examples(golden_path)
    service = RagService()

    report = {
        "naive": _score(
            examples,
            lambda question: service.retrieve_naive(question, top_k=10),
        ),
        "advanced": _score(
            examples,
            lambda question: service.retrieve_advanced(question, top_k=10)[0],
        ),
    }
    output_path = ROOT / "eval_report.json"
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

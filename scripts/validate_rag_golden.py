#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "evals" / "rag_golden.jsonl"
RAG_STORE_PATH = ROOT / "data" / "rag_store.json"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_rag_chunk_ids() -> set[str]:
    if not RAG_STORE_PATH.exists():
        return set()
    data = json.loads(RAG_STORE_PATH.read_text(encoding="utf-8"))
    return {
        str(chunk.get("id"))
        for chunk in data.get("chunks", [])
        if isinstance(chunk, dict) and chunk.get("id")
    }


def main() -> None:
    if not GOLDEN_PATH.exists():
        fail(f"Missing RAG golden file: {GOLDEN_PATH.relative_to(ROOT)}")

    rows = []
    for line_number, line in enumerate(GOLDEN_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            fail(f"Invalid JSON on line {line_number}: {exc}")

    if len(rows) != 25:
        fail(f"Expected exactly 25 rows, found {len(rows)}")

    known_chunk_ids = load_rag_chunk_ids()
    missing_chunks: list[tuple[int, str]] = []

    for index, row in enumerate(rows, start=1):
        if not str(row.get("question") or "").strip():
            fail(f"Row {index} is missing question")
        if not str(row.get("ideal_answer") or "").strip():
            fail(f"Row {index} is missing ideal_answer")
        chunk_ids = row.get("ground_truth_chunk_ids")
        if not isinstance(chunk_ids, list) or not chunk_ids:
            fail(f"Row {index} must have non-empty ground_truth_chunk_ids")
        for chunk_id in chunk_ids:
            chunk_id_text = str(chunk_id)
            if chunk_id_text.startswith("placeholder"):
                fail(f"Row {index} has placeholder chunk id: {chunk_id_text}")
            if known_chunk_ids and chunk_id_text not in known_chunk_ids:
                missing_chunks.append((index, chunk_id_text))

    if missing_chunks:
        details = ", ".join(f"row {row}: {chunk_id}" for row, chunk_id in missing_chunks)
        fail(f"Chunk IDs not found in {RAG_STORE_PATH.relative_to(ROOT)}: {details}")

    print("RAG golden file is valid.")


if __name__ == "__main__":
    main()

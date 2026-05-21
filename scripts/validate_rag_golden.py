#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "evals" / "rag_golden_set.jsonl"
PROCESSED_CHUNKS_PATH = ROOT / "data" / "rag" / "processed" / "rag_chunks.jsonl"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_rag_keys() -> set[str]:
    if not PROCESSED_CHUNKS_PATH.exists():
        return set()
    keys = set()
    for line in PROCESSED_CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = json.loads(line)
        keys.add(str(chunk.get("id") or ""))
        keys.add(str(chunk.get("source_id") or ""))
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        keys.add(str(metadata.get("path") or ""))
    return {key for key in keys if key}


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

    known_keys = load_rag_keys()
    missing_chunks: list[tuple[int, str]] = []

    for index, row in enumerate(rows, start=1):
        if not str(row.get("question") or "").strip():
            fail(f"Row {index} is missing question")
        if not str(row.get("ideal_answer") or "").strip():
            fail(f"Row {index} is missing ideal_answer")
        chunk_ids = row.get("ground_truth_chunk_ids") or row.get("ground_truth_chunks")
        if not isinstance(chunk_ids, list) or not chunk_ids:
            fail(f"Row {index} must have non-empty ground_truth_chunks")
        for chunk_id in chunk_ids:
            chunk_id_text = str(chunk_id)
            if chunk_id_text.startswith("placeholder"):
                fail(f"Row {index} has placeholder chunk id: {chunk_id_text}")
            if known_keys and chunk_id_text not in known_keys:
                missing_chunks.append((index, chunk_id_text))

    if missing_chunks:
        details = ", ".join(f"row {row}: {chunk_id}" for row, chunk_id in missing_chunks)
        fail(f"Ground-truth sources not found in {PROCESSED_CHUNKS_PATH.relative_to(ROOT)}: {details}")

    print("RAG golden file is valid.")


if __name__ == "__main__":
    main()

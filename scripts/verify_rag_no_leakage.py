#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "backend" / "model_server" / "artifacts"
CLASSIFIER_TRAIN_PATH = ARTIFACT_DIR / "train.csv"
HELDOUT_RAG_PATH = ROOT / "data" / "rag" / "heldout_resolved_issues.jsonl"
FINAL_RAG_PATH = ROOT / "data" / "rag" / "final_issues" / "node_resolved_issues.jsonl"
RAW_RESOLVED_PATH = ROOT / "data" / "raw" / "resolved_issues_raw.json"
NUMBER_KEYS = ("issue_number", "number", "issue_id", "id")


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def normalize_issue_number(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows = []
        for key in ("issues", "documents", "rows", "data"):
            value = data.get(key)
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
        if rows:
            return rows
        return [data]
    return []


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"Invalid JSONL in {path.relative_to(ROOT)} line {line_number}: {exc}")
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_rows(path)
    if suffix == ".jsonl":
        return load_jsonl_rows(path)
    if suffix == ".json":
        return load_json_rows(path)
    return []


def extract_issue_numbers(rows: list[dict[str, Any]]) -> set[int]:
    numbers = set()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        candidates = [row.get(key) for key in NUMBER_KEYS]
        candidates.extend(metadata.get(key) for key in NUMBER_KEYS)
        for candidate in candidates:
            issue_number = normalize_issue_number(candidate)
            if issue_number is not None:
                numbers.add(issue_number)
                break
    return numbers


def load_classifier_train_issue_numbers() -> set[int]:
    numbers = set()
    if not CLASSIFIER_TRAIN_PATH.exists():
        fail(f"Missing classifier train split: {CLASSIFIER_TRAIN_PATH.relative_to(ROOT)}")
    for row in load_csv_rows(CLASSIFIER_TRAIN_PATH):
        issue_number = normalize_issue_number(row.get("issue_number") or row.get("number"))
        if issue_number is not None:
            numbers.add(issue_number)
    return numbers


def main() -> None:
    classifier_numbers = load_classifier_train_issue_numbers()
    if not classifier_numbers:
        fail("No classifier train issue numbers found")

    if not HELDOUT_RAG_PATH.exists():
        fail(
            f"Missing clean held-out RAG file: {HELDOUT_RAG_PATH.relative_to(ROOT)}. "
            "Run python3 scripts/build_clean_rag_issues.py first."
        )

    final_numbers = extract_issue_numbers(load_rows(FINAL_RAG_PATH)) if FINAL_RAG_PATH.exists() else set()
    final_overlaps = sorted(final_numbers & classifier_numbers)
    if final_overlaps:
        print(f"RAG/classifier leakage detected in {FINAL_RAG_PATH.relative_to(ROOT)}:")
        for issue_number in final_overlaps:
            print(f"  issue {issue_number}")
        raise SystemExit(1)

    legacy_numbers = extract_issue_numbers(load_rows(HELDOUT_RAG_PATH))
    legacy_overlaps = sorted(legacy_numbers & classifier_numbers)
    if legacy_overlaps:
        preview = ", ".join(str(number) for number in legacy_overlaps[:20])
        suffix = "..." if len(legacy_overlaps) > 20 else ""
        print(
            f"WARNING: legacy demo RAG file {HELDOUT_RAG_PATH.relative_to(ROOT)} overlaps "
            f"classifier training ({len(legacy_overlaps)} issues): {preview}{suffix}"
        )

    checked_count = len(final_numbers) if final_numbers else len(legacy_numbers)
    if not final_numbers and legacy_overlaps:
        print(f"RAG/classifier leakage detected in {HELDOUT_RAG_PATH.relative_to(ROOT)}:")
        for issue_number in legacy_overlaps:
            print(f"  issue {issue_number}")
        raise SystemExit(1)

    print(
        f"No RAG/classifier issue-number overlap found in "
        f"production RAG issue file ({checked_count} issues checked)."
    )

    if RAW_RESOLVED_PATH.exists():
        raw_numbers = extract_issue_numbers(load_rows(RAW_RESOLVED_PATH))
        raw_overlaps = sorted(raw_numbers & classifier_numbers)
        if raw_overlaps:
            preview = ", ".join(str(number) for number in raw_overlaps[:20])
            suffix = "..." if len(raw_overlaps) > 20 else ""
            print(
                f"WARNING: Raw input {RAW_RESOLVED_PATH.relative_to(ROOT)} still overlaps "
                f"classifier splits ({len(raw_overlaps)} issues): {preview}{suffix}"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden" / "classification_golden.jsonl"
VALID_LABELS = {"bug", "feature", "docs", "question"}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not GOLDEN_PATH.exists():
        fail(f"Missing golden file: {GOLDEN_PATH.relative_to(ROOT)}")

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

    issue_numbers = []
    label_counts = Counter()
    for index, row in enumerate(rows, start=1):
        label = row.get("expected_label")
        if label not in VALID_LABELS:
            fail(f"Row {index} has invalid expected_label: {label!r}")

        issue_number = row.get("issue_number")
        if issue_number not in (None, ""):
            issue_numbers.append(str(issue_number))

        text = str(row.get("text") or "").strip()
        title = str(row.get("title") or "").strip()
        body = str(row.get("body") or "").strip()
        if not text and not (title and body):
            fail(f"Row {index} must have text or title/body")

        label_counts[label] += 1

    duplicates = sorted(number for number, count in Counter(issue_numbers).items() if count > 1)
    if duplicates:
        fail(f"Duplicate issue_number values: {', '.join(duplicates)}")

    print("classification golden label counts:")
    for label in sorted(VALID_LABELS):
        print(f"  {label}: {label_counts[label]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "backend" / "model_server" / "artifacts"
SPLIT_PATHS = [
    ARTIFACT_DIR / "train.csv",
    ARTIFACT_DIR / "val.csv",
    ARTIFACT_DIR / "test.csv",
]
SOURCE_PATH = ROOT / "data" / "raw" / "resolved_issues_raw.json"
OUTPUT_PATH = ROOT / "data" / "rag" / "heldout_resolved_issues.jsonl"
MANIFEST_PATH = ROOT / "data" / "rag" / "heldout_manifest.json"


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def issue_number(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return None


def load_classifier_issue_numbers() -> set[int]:
    numbers: set[int] = set()
    for split_path in SPLIT_PATHS:
        if not split_path.exists():
            fail(f"Missing classifier split file: {rel(split_path)}")
        with split_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "issue_number" not in (reader.fieldnames or []):
                fail(f"{rel(split_path)} is missing required column issue_number")
            for row in reader:
                number = issue_number(row.get("issue_number"))
                if number is not None:
                    numbers.add(number)
    return numbers


def load_source_issues() -> list[dict[str, Any]]:
    if not SOURCE_PATH.exists():
        fail(f"Missing source RAG issue file: {rel(SOURCE_PATH)}")
    try:
        data = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{rel(SOURCE_PATH)} is not valid JSON: {exc}")
    if not isinstance(data, list):
        fail(f"{rel(SOURCE_PATH)} must contain a JSON array of issues")
    return [item for item in data if isinstance(item, dict)]


def normalize_issue(row: dict[str, Any]) -> dict[str, Any] | None:
    number = issue_number(row.get("number") or row.get("issue_number"))
    if number is None:
        return None
    return {
        "issue_number": number,
        "title": row.get("title") or "",
        "body": row.get("body") or "",
        "url": row.get("url"),
        "created_at": row.get("created_at") or row.get("createdAt"),
        "closed_at": row.get("closed_at") or row.get("closedAt"),
        "source_type": "issue",
    }


def main() -> None:
    classifier_numbers = load_classifier_issue_numbers()
    source_issues = load_source_issues()

    kept: list[dict[str, Any]] = []
    removed: list[int] = []

    for row in source_issues:
        normalized = normalize_issue(row)
        if normalized is None:
            continue
        number = normalized["issue_number"]
        if number in classifier_numbers:
            removed.append(number)
            continue
        kept.append(normalized)

    removed = sorted(set(removed))
    if not kept:
        fail("No held-out RAG issues remain after removing classifier split overlaps")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )

    manifest = {
        "source_file": rel(SOURCE_PATH),
        "output_file": rel(OUTPUT_PATH),
        "total_input_issues": len(source_issues),
        "classifier_split_issue_count": len(classifier_numbers),
        "removed_overlap_count": len(removed),
        "kept_heldout_count": len(kept),
        "removed_issue_numbers": removed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("Clean RAG held-out issue corpus built.")
    print(f"  Source issues: {len(source_issues)}")
    print(f"  Classifier issue numbers: {len(classifier_numbers)}")
    print(f"  Removed overlaps: {len(removed)}")
    print(f"  Kept held-out issues: {len(kept)}")
    print(f"  Wrote: {rel(OUTPUT_PATH)}")
    print(f"  Manifest: {rel(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()

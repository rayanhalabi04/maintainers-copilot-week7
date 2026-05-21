#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "backend" / "model_server" / "artifacts" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "rag" / "final_issues" / "node_resolved_issues.jsonl"
RAW_INPUT_PATH = ROOT / "data" / "raw" / "resolved_issues_raw.json"
REPO = "nodejs/node"
MAINTAINER_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch closed Node.js issues with maintainer comments for RAG."
    )
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--exclude-train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--raw-input", type=Path, default=RAW_INPUT_PATH)
    parser.add_argument("--target-count", type=int, default=250)
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--per-page", type=int, default=100)
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        if args.raw_input.exists():
            records = build_records_from_raw(args.raw_input, args.repo, load_issue_numbers(args.exclude_train), args.target_count)
            write_records(args.output, records, args.repo, args.target_count, "local_raw")
            return
        print(
            "ERROR: GITHUB_TOKEN is required unless --raw-input points to an existing raw issues JSON file.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    excluded = load_issue_numbers(args.exclude_train)
    records = []
    skipped_open = 0
    skipped_train = 0
    skipped_no_maintainer = 0

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "maintainers-copilot-rag-ingest",
        }
    )

    for page in range(1, args.pages + 1):
        issues = github_get(
            session,
            f"https://api.github.com/repos/{args.repo}/issues",
            params={
                "state": "closed",
                "sort": "updated",
                "direction": "desc",
                "per_page": args.per_page,
                "page": page,
            },
        )
        if not issues:
            break
        for issue in issues:
            if "pull_request" in issue:
                continue
            number = int(issue["number"])
            if issue.get("state") != "closed" or not issue.get("closed_at"):
                skipped_open += 1
                continue
            if number in excluded:
                skipped_train += 1
                continue

            comments = github_get(session, issue["comments_url"])
            maintainer_comments = [
                normalize_comment(comment)
                for comment in comments
                if comment.get("author_association") in MAINTAINER_ASSOCIATIONS
                and str(comment.get("body") or "").strip()
            ]
            if not maintainer_comments:
                skipped_no_maintainer += 1
                continue

            labels = [
                label.get("name")
                for label in issue.get("labels", [])
                if isinstance(label, dict) and label.get("name")
            ]
            records.append(
                {
                    "source_type": "issue",
                    "source_id": f"node-issue-{number}",
                    "number": number,
                    "issue_number": number,
                    "title": issue.get("title") or f"Node.js issue {number}",
                    "body": issue.get("body") or "",
                    "maintainer_comments": maintainer_comments[:5],
                    "maintainer_answer": select_maintainer_answer(maintainer_comments),
                    "labels": labels,
                    "url": issue.get("html_url"),
                    "created_at": issue.get("created_at"),
                    "closed_at": issue.get("closed_at"),
                    "metadata": {
                        "repo": args.repo,
                        "quality": "maintainer_comment",
                        "comment_count": len(comments),
                        "maintainer_comment_count": len(maintainer_comments),
                    },
                }
            )
            if len(records) >= args.target_count:
                break
        if len(records) >= args.target_count:
            break

    write_records(
        args.output,
        records,
        args.repo,
        args.target_count,
        "github_api",
        {
            "excluded_train_issue_count": len(excluded),
            "skipped_train": skipped_train,
            "skipped_open": skipped_open,
            "skipped_no_maintainer": skipped_no_maintainer,
        },
    )
    if len(records) == 0:
        raise SystemExit(1)


def build_records_from_raw(
    raw_input: Path,
    repo: str,
    excluded: set[int],
    target_count: int,
) -> list[dict[str, Any]]:
    data = json.loads(raw_input.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{raw_input.relative_to(ROOT)} must contain a JSON array.")
    records = []
    for issue in data:
        if not isinstance(issue, dict):
            continue
        number = issue.get("number") or issue.get("issue_number")
        if not isinstance(number, int):
            continue
        closed_at = issue.get("closed_at") or issue.get("closedAt")
        if not closed_at or number in excluded:
            continue
        comments = issue.get("comments") or []
        maintainer_comments = []
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            association = comment.get("author_association") or comment.get("authorAssociation")
            body = str(comment.get("body") or "").strip()
            if association in MAINTAINER_ASSOCIATIONS and body:
                maintainer_comments.append(
                    {
                        "author": (comment.get("author") or comment.get("user") or {}).get("login"),
                        "author_association": association,
                        "body": body,
                        "created_at": comment.get("created_at") or comment.get("createdAt"),
                        "url": comment.get("html_url") or comment.get("url"),
                    }
                )
        if not maintainer_comments:
            continue
        labels = []
        for label in issue.get("labels") or []:
            if isinstance(label, dict) and label.get("name"):
                labels.append(label["name"])
            elif isinstance(label, str):
                labels.append(label)
        records.append(
            {
                "source_type": "issue",
                "source_id": f"node-issue-{number}",
                "number": number,
                "issue_number": number,
                "title": issue.get("title") or f"Node.js issue {number}",
                "body": issue.get("body") or "",
                "maintainer_comments": maintainer_comments[:5],
                "maintainer_answer": select_maintainer_answer(maintainer_comments),
                "labels": labels,
                "url": issue.get("html_url") or issue.get("url"),
                "created_at": issue.get("created_at") or issue.get("createdAt"),
                "closed_at": closed_at,
                "metadata": {
                    "repo": repo,
                    "quality": "maintainer_comment",
                    "source_file": str(raw_input.relative_to(ROOT)),
                    "maintainer_comment_count": len(maintainer_comments),
                },
            }
        )
        if len(records) >= target_count:
            break
    return records


def write_records(
    output: Path,
    records: list[dict[str, Any]],
    repo: str,
    target_count: int,
    mode: str,
    extra_manifest: dict[str, Any] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    manifest = {
        "repo": repo,
        "mode": mode,
        "output": str(output.relative_to(ROOT)),
        "target_count": target_count,
        "records": len(records),
        "maintainer_associations": sorted(MAINTAINER_ASSOCIATIONS),
    }
    manifest.update(extra_manifest or {})
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} held-out closed Node.js issues to {output.relative_to(ROOT)}.")


def github_get(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> Any:
    response = session.get(url, params=params, timeout=30)
    if response.status_code == 403:
        raise SystemExit(f"GitHub API rejected the request: {response.text[:300]}")
    response.raise_for_status()
    return response.json()


def load_issue_numbers(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        numbers = set()
        for row in reader:
            number = row.get("issue_number") or row.get("number")
            if number and str(number).isdigit():
                numbers.add(int(number))
        return numbers


def normalize_comment(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "author": (comment.get("user") or {}).get("login"),
        "author_association": comment.get("author_association"),
        "body": str(comment.get("body") or "").strip(),
        "created_at": comment.get("created_at"),
        "url": comment.get("html_url"),
    }


def select_maintainer_answer(comments: list[dict[str, Any]]) -> str:
    closing_words = ("fixed", "resolved", "duplicate", "closing", "landed", "answered")
    for comment in reversed(comments):
        body = str(comment.get("body") or "")
        if any(word in body.lower() for word in closing_words):
            return body
    return str(comments[-1].get("body") or "")


if __name__ == "__main__":
    main()

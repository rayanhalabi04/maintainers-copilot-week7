import json
import time
from pathlib import Path

import requests

OWNER = "pandas-dev"
REPO = "pandas"
OUTPUT_PATH = Path("data/raw/issues.jsonl")

LABEL_TO_TARGET = {
    "bug": "bug",
    "enhancement": "feature",
    "docs": "docs",
    "usage question": "question",
}

MAX_PER_LABEL = 2000


def fetch_label(label: str) -> list[dict]:
    page = 1
    results = []

    while len(results) < MAX_PER_LABEL:
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
        params = {
            "state": "closed",
            "labels": label,
            "per_page": 100,
            "page": page,
            "sort": "created",
            "direction": "asc",
        }

        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 403:
            print("Rate limited by GitHub. Waiting 60 seconds...")
            time.sleep(60)
            continue

        if response.status_code == 422:
            print(f"GitHub returned 422 for label={label}. Stopping this label.")
            break

        response.raise_for_status()
        issues = response.json()

        if not issues:
            break

        for issue in issues:
            if "pull_request" in issue:
                continue

            results.append(issue)

            if len(results) >= MAX_PER_LABEL:
                break

        print(f"Label {label} page {page} done. Collected: {len(results)}")
        page += 1

    return results


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set()
    saved = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for github_label, target_label in LABEL_TO_TARGET.items():
            issues = fetch_label(github_label)

            for issue in issues:
                if issue["id"] in seen_ids:
                    continue

                seen_ids.add(issue["id"])

                labels = [label["name"].lower() for label in issue.get("labels", [])]

                record = {
                    "id": issue["id"],
                    "number": issue["number"],
                    "title": issue.get("title") or "",
                    "body": issue.get("body") or "",
                    "labels": labels,
                    "target_label": target_label,
                    "created_at": issue.get("created_at"),
                    "closed_at": issue.get("closed_at"),
                    "url": issue.get("html_url"),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1

    print(f"Done. Saved {saved} issues to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "rag" / "final_docs"
LOCAL_NODE_DIR = ROOT / "external" / "node"
REPO = "nodejs/node"
REF = os.getenv("NODE_DOCS_REF", "main")

ROOT_DOCS = [
    "README.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "BUILDING.md",
    "GOVERNANCE.md",
]
API_DOCS = [
    "doc/api/assert.md",
    "doc/api/buffer.md",
    "doc/api/cli.md",
    "doc/api/crypto.md",
    "doc/api/errors.md",
    "doc/api/fs.md",
    "doc/api/http.md",
    "doc/api/modules.md",
    "doc/api/process.md",
    "doc/api/stream.md",
    "doc/api/test.md",
    "doc/api/url.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy or fetch selected real Node.js documentation for RAG."
    )
    parser.add_argument("--source-dir", type=Path, default=LOCAL_NODE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ref", default=REF)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    wanted_paths = ROOT_DOCS + API_DOCS
    copied = 0
    downloaded = 0
    missing = []

    for rel_path in wanted_paths:
        target = args.output_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        local_source = args.source_dir / rel_path

        if local_source.exists() and not args.force_download:
            shutil.copyfile(local_source, target)
            copied += 1
            continue

        try:
            content = fetch_raw_doc(rel_path, args.ref)
        except (HTTPError, URLError, TimeoutError) as exc:
            missing.append(f"{rel_path}: {exc}")
            continue
        target.write_bytes(content)
        downloaded += 1

    manifest = args.output_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"repo={REPO}",
                f"ref={args.ref}",
                f"copied={copied}",
                f"downloaded={downloaded}",
                "paths=",
                *wanted_paths,
                "missing=",
                *missing,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if copied + downloaded == 0:
        print("ERROR: no Node.js docs were copied or downloaded.", file=sys.stderr)
        raise SystemExit(1)
    if missing:
        print("WARNING: some docs could not be fetched:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
    print(
        f"Prepared {copied + downloaded} Node.js doc files in "
        f"{args.output_dir.relative_to(ROOT)}."
    )


def fetch_raw_doc(path: str, ref: str) -> bytes:
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{path}"
    headers = {"User-Agent": "maintainers-copilot-rag-ingest"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        return response.read()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.domain.rag import RagDocument  # noqa: E402
from app.services.rag_chunking import ParentChildChunker, stable_id  # noqa: E402


DOCS_DIR = ROOT / "data" / "rag" / "final_docs"
ISSUES_PATH = ROOT / "data" / "rag" / "final_issues" / "node_resolved_issues.jsonl"
PROCESSED_DIR = ROOT / "data" / "rag" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized Node.js RAG documents and debug chunks."
    )
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--issues-file", type=Path, default=ISSUES_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    documents = list(load_docs(args.docs_dir))
    documents.extend(load_issues(args.issues_file))
    if not documents:
        raise SystemExit("No docs or issues found. Run fetch_node_docs.py and fetch_node_resolved_issues.py first.")

    chunker = ParentChildChunker()
    chunks = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    docs_path = args.output_dir / "rag_documents.jsonl"
    chunks_path = args.output_dir / "rag_chunks.jsonl"
    docs_path.write_text(
        "".join(doc.model_dump_json() + "\n" for doc in documents),
        encoding="utf-8",
    )
    chunks_path.write_text(
        "".join(chunk.model_dump_json() + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    manifest = {
        "repo": "nodejs/node",
        "documents": len(documents),
        "chunks": len(chunks),
        "doc_documents": sum(1 for doc in documents if doc.source_type == "doc"),
        "issue_documents": sum(1 for doc in documents if doc.source_type == "issue"),
        "docs_path": str(docs_path.relative_to(ROOT)),
        "chunks_path": str(chunks_path.relative_to(ROOT)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def load_docs(docs_dir: Path):
    if not docs_dir.exists():
        return
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir)
        text = path.read_text(encoding="utf-8", errors="replace")
        yield RagDocument(
            id=stable_id("nodejs/node", "doc", str(rel)),
            source_type="doc",
            source_id=str(rel),
            title=title_from_path(rel, text),
            url=f"https://github.com/nodejs/node/blob/main/{rel.as_posix()}",
            text=text,
            metadata={
                "repo": "nodejs/node",
                "path": rel.as_posix(),
                "source_type": "doc",
                "corpus": "nodejs-node-final",
            },
        )


def load_issues(path: Path) -> list[RagDocument]:
    if not path.exists():
        return []
    documents = []
    for row in read_jsonl(path):
        number = row.get("number") or row.get("issue_number")
        source_id = row.get("source_id") or f"node-issue-{number}"
        title = str(row.get("title") or f"Node.js issue {number}")
        body = str(row.get("body") or "")
        answer = str(row.get("maintainer_answer") or "")
        comments = row.get("maintainer_comments") or []
        comment_text = "\n\n".join(
            f"Maintainer comment by {comment.get('author')}: {comment.get('body')}"
            for comment in comments
            if isinstance(comment, dict) and comment.get("body")
        )
        text = "\n\n".join(
            part
            for part in [
                f"# {title}",
                "## Issue body\n" + body if body else "",
                "## Selected maintainer answer\n" + answer if answer else "",
                "## Maintainer comments\n" + comment_text if comment_text else "",
            ]
            if part
        )
        metadata = {
            **(row.get("metadata") or {}),
            "repo": "nodejs/node",
            "labels": row.get("labels") or [],
            "created_at": row.get("created_at"),
            "closed_at": row.get("closed_at"),
            "resolved_at": row.get("closed_at"),
            "issue_number": number,
            "number": number,
            "source_type": "issue",
            "corpus": "nodejs-node-final",
        }
        documents.append(
            RagDocument(
                id=stable_id("nodejs/node", "issue", source_id),
                source_type="issue",
                source_id=str(source_id),
                title=title,
                url=row.get("url"),
                text=text,
                metadata=metadata,
            )
        )
    return documents


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def title_from_path(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    if path.name == "README.md":
        return "Node.js README"
    return path.stem.replace("-", " ").replace("_", " ").title()


if __name__ == "__main__":
    main()

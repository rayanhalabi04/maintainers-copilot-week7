import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.domain.rag import RagDocument  # noqa: E402
from app.infra.embedding_provider import SentenceTransformerEmbeddingProvider  # noqa: E402
from app.infra.rag_config import RagConfig  # noqa: E402
from app.repositories.rag_repository import create_rag_repository  # noqa: E402
from app.services.rag_chunking import ParentChildChunker, stable_id  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--issues-file", default="data/rag/heldout_resolved_issues.jsonl")
    args = parser.parse_args()

    config = RagConfig()
    repository = create_rag_repository(
        config.database_url,
        config.local_store_path,
        force_local_store=config.force_local_store,
    )
    embedder = SentenceTransformerEmbeddingProvider(config.embedding_model)
    chunker = ParentChildChunker()

    documents = list(load_docs(Path(args.docs_dir)))
    if not documents:
        documents = list(load_docs(Path("data/rag/sample_docs")))
    documents.extend(load_issues(Path(args.issues_file)))

    total_chunks = 0
    for document in documents:
        chunks = chunker.chunk_document(document)
        embeddings = embedder.embed([chunk.text for chunk in chunks])
        chunks = [
            chunk.model_copy(update={"embedding": embedding})
            for chunk, embedding in zip(chunks, embeddings)
        ]
        repository.upsert_document_with_chunks(document, chunks)
        total_chunks += len(chunks)

    print(f"Ingested {len(documents)} parent documents and {total_chunks} child chunks.")


def load_docs(docs_dir: Path):
    if not docs_dir.exists():
        return
    for path in sorted(docs_dir.rglob("*.md")):
        text = path.read_text()
        rel = path.relative_to(docs_dir)
        yield RagDocument(
            id=stable_id("doc", str(rel)),
            source_type="doc",
            source_id=str(rel),
            title=path.stem.replace("-", " ").replace("_", " ").title(),
            url=None,
            text=text,
            metadata={"path": str(path), "source_type": "doc"},
        )


def load_issues(path: Path):
    if not path.exists():
        return []
    rows = read_rows(path)
    documents = []
    for row in rows:
        issue_number = row.get("issue_number") or row.get("number")
        source_id = str(issue_number or row.get("id"))
        title = str(row.get("title") or f"Issue {source_id}")
        comments = row.get("comments") or []
        if isinstance(comments, str):
            comments = [comments]
        body_parts = [str(row.get("body") or "")]
        body_parts.extend(str(comment) for comment in comments)
        text = "\n\n".join(part for part in body_parts if part)
        labels = row.get("labels") or []
        if isinstance(labels, str):
            labels = [item.strip() for item in labels.split(",") if item.strip()]
        metadata = {
            "labels": labels,
            "created_at": row.get("created_at"),
            "resolved_at": row.get("closed_at") or row.get("resolved_at"),
            "issue_number": issue_number,
            "number": issue_number,
            "source_type": "issue",
        }
        documents.append(
            RagDocument(
                id=stable_id("issue", source_id),
                source_type="issue",
                source_id=source_id,
                title=title,
                url=row.get("url"),
                text=text,
                metadata=metadata,
            )
        )
    return documents


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    main()

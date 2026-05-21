import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.rag import RagChunk, RagDocument, RagFilters
from app.repositories.rag_repository import LocalJsonRagRepository, PostgresRagRepository, create_rag_repository
from app.services.rag_chunking import ParentChildChunker
from scripts.build_final_rag_corpus import load_docs, load_issues
from scripts.fetch_node_resolved_issues import load_issue_numbers


def test_production_repository_uses_pgvector_when_database_url_is_set(tmp_path: Path):
    repository = create_rag_repository(
        "postgresql://maintainers:maintainers@localhost:5432/maintainers_copilot",
        str(tmp_path / "rag_store.json"),
        force_local_store=False,
    )

    assert isinstance(repository, PostgresRagRepository)


def test_dev_repository_can_still_use_local_json_fallback(tmp_path: Path):
    repository = create_rag_repository(
        "postgresql://maintainers:maintainers@localhost:5432/maintainers_copilot",
        str(tmp_path / "rag_store.json"),
        force_local_store=True,
    )

    assert isinstance(repository, LocalJsonRagRepository)


def test_final_corpus_loaders_create_node_docs_and_issues(tmp_path: Path):
    docs_dir = tmp_path / "final_docs"
    docs_dir.mkdir()
    (docs_dir / "README.md").write_text("# Node.js README\n\nDownload Node.js from nodejs.org.\n")
    issues_path = tmp_path / "node_resolved_issues.jsonl"
    issues_path.write_text(
        json.dumps(
            {
                "source_type": "issue",
                "source_id": "node-issue-40306",
                "number": 40306,
                "title": "fs error on Windows",
                "body": "A user reported an fs failure.",
                "maintainer_answer": "A collaborator explained the expected behavior.",
                "maintainer_comments": [
                    {
                        "author": "node-maintainer",
                        "author_association": "MEMBER",
                        "body": "This was resolved by updating the test.",
                    }
                ],
                "labels": ["fs"],
                "url": "https://github.com/nodejs/node/issues/40306",
                "closed_at": "2024-01-01T00:00:00Z",
                "metadata": {"repo": "nodejs/node"},
            }
        )
        + "\n"
    )

    documents = list(load_docs(docs_dir))
    documents.extend(load_issues(issues_path))

    assert {document.source_type for document in documents} == {"doc", "issue"}
    assert all(document.metadata["repo"] == "nodejs/node" for document in documents)
    assert documents[0].url == "https://github.com/nodejs/node/blob/main/README.md"
    assert documents[1].url == "https://github.com/nodejs/node/issues/40306"


def test_train_issue_numbers_are_loaded_for_rag_exclusion(tmp_path: Path):
    train_path = tmp_path / "train.csv"
    with train_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["issue_number", "title"])
        writer.writeheader()
        writer.writerow({"issue_number": "40306", "title": "training issue"})

    assert load_issue_numbers(train_path) == {40306}


def test_local_rag_retrieval_returns_node_sources_and_filters_metadata(tmp_path: Path):
    repository = LocalJsonRagRepository(str(tmp_path / "rag_store.json"))
    document = RagDocument(
        id="doc-node-readme",
        source_type="doc",
        source_id="README.md",
        title="Node.js README",
        text="# Node.js README\n\nNode.js is a JavaScript runtime.",
        url="https://github.com/nodejs/node/blob/main/README.md",
        metadata={"repo": "nodejs/node", "path": "README.md"},
    )
    chunk = RagChunk(
        id="chunk-node-readme",
        document_id=document.id,
        source_type="doc",
        source_id=document.source_id,
        title=document.title,
        text="Node.js is a JavaScript runtime.",
        chunk_index=0,
        url=document.url,
        embedding=[1.0, 0.0],
        metadata={"repo": "nodejs/node", "path": "README.md", "chunk_index": 0},
    )
    sample_document = RagDocument(
        id="doc-sample",
        source_type="doc",
        source_id="sample_docs/login.md",
        title="Sample Login",
        text="Sample JWT text.",
        metadata={"repo": "example/demo", "path": "data/rag/sample_docs/login.md"},
    )
    sample_chunk = RagChunk(
        id="chunk-sample",
        document_id=sample_document.id,
        source_type="doc",
        source_id=sample_document.source_id,
        title=sample_document.title,
        text="Sample JWT text.",
        chunk_index=0,
        embedding=[0.0, 1.0],
        metadata={"repo": "example/demo", "path": "data/rag/sample_docs/login.md", "chunk_index": 0},
    )
    repository.upsert_document_with_chunks(document, [chunk])
    repository.upsert_document_with_chunks(sample_document, [sample_chunk])

    results = repository.dense_search([1.0, 0.0], 5, RagFilters(repo="nodejs/node", path="README.md"))

    assert len(results) == 1
    assert results[0].url == "https://github.com/nodejs/node/blob/main/README.md"
    assert results[0].metadata["repo"] == "nodejs/node"
    assert "sample_docs/login.md" not in results[0].source_id


def test_issue_chunking_keeps_short_body_with_maintainer_answer():
    document = RagDocument(
        id="issue-node-1",
        source_type="issue",
        source_id="node-issue-1",
        title="Node issue",
        text=(
            "# Node issue\n\n"
            "## Issue body\n"
            "The fs test fails on Windows.\n\n"
            "## Selected maintainer answer\n"
            "A maintainer says the fix is to update the assertion."
        ),
        metadata={"repo": "nodejs/node", "labels": ["fs"]},
    )

    chunks = ParentChildChunker().chunk_document(document)

    assert any(
        "The fs test fails" in chunk.text and "update the assertion" in chunk.text
        for chunk in chunks
    )

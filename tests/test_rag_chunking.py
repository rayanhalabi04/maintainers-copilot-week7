from app.domain.rag import RagDocument
from app.services.rag_chunking import ParentChildChunker


def test_parent_child_chunking_preserves_document_reference():
    document = RagDocument(
        id="doc-1",
        source_type="doc",
        source_id="guide.md",
        title="Guide",
        text="# Guide\n\n## Install\n\nRun `uv sync`.\n\n## Login\n\nClear stale JWT tokens.",
    )

    chunks = ParentChildChunker().chunk_document(document)

    assert chunks
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
    assert any("Login" in chunk.text or "JWT" in chunk.text for chunk in chunks)

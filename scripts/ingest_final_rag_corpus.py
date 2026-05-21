#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.domain.rag import RagDocument  # noqa: E402
from app.infra.embedding_provider import SentenceTransformerEmbeddingProvider  # noqa: E402
from app.infra.rag_config import RagConfig  # noqa: E402
from app.repositories.rag_repository import create_rag_repository  # noqa: E402
from app.services.rag_chunking import ParentChildChunker  # noqa: E402


DOCS_PATH = ROOT / "data" / "rag" / "processed" / "rag_documents.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and ingest final RAG corpus into pgvector or dev JSON store.")
    parser.add_argument("--documents", type=Path, default=DOCS_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.documents.exists():
        raise SystemExit(f"Missing {args.documents}. Run scripts/build_final_rag_corpus.py first.")

    config = RagConfig()
    repository = create_rag_repository(
        config.database_url,
        config.local_store_path,
        force_local_store=config.force_local_store,
    )
    embedder = SentenceTransformerEmbeddingProvider(config.embedding_model)
    chunker = ParentChildChunker()

    documents = [
        RagDocument(**json.loads(line))
        for line in args.documents.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    total_chunks = 0
    for document in documents:
        chunks = chunker.chunk_document(document)
        embedded_chunks = []
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start : start + args.batch_size]
            embeddings = embedder.embed([chunk.text for chunk in batch])
            embedded_chunks.extend(
                chunk.model_copy(update={"embedding": embedding})
                for chunk, embedding in zip(batch, embeddings)
            )
        repository.upsert_document_with_chunks(document, embedded_chunks)
        total_chunks += len(embedded_chunks)

    print(f"Ingested {len(documents)} Node.js documents and {total_chunks} chunks.")
    print(f"Repository: {repository.__class__.__name__}")


if __name__ == "__main__":
    main()

import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from app.domain.rag import RagChunk, RagDocument, RagFilters, RetrievedChunk


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return max(0.0, dot / (left_norm * right_norm))


def _tokens(text: str) -> set[str]:
    return {token.strip(".,:;()[]{}<>`'\"").lower() for token in text.split() if token.strip()}


def _token_list(text: str) -> list[str]:
    return [
        token.strip(".,:;()[]{}<>`'\"").lower()
        for token in text.split()
        if token.strip(".,:;()[]{}<>`'\"")
    ]


class RagRepository:
    def upsert_document_with_chunks(self, document: RagDocument, chunks: list[RagChunk]) -> None:
        raise NotImplementedError

    def dense_search(
        self, query_embedding: list[float], top_k: int, filters: RagFilters | None
    ) -> list[RetrievedChunk]:
        raise NotImplementedError

    def sparse_search(self, query: str, top_k: int, filters: RagFilters | None) -> list[RetrievedChunk]:
        raise NotImplementedError

    def get_sibling_chunks(self, document_id: str, chunk_index: int, window: int = 1) -> list[RagChunk]:
        return []


class LocalJsonRagRepository(RagRepository):
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.documents: dict[str, RagDocument] = {}
        self.chunks: dict[str, RagChunk] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text())
        self.documents = {
            item["id"]: RagDocument(**item) for item in data.get("documents", [])
        }
        self.chunks = {item["id"]: RagChunk(**item) for item in data.get("chunks", [])}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": [doc.model_dump(mode="json") for doc in self.documents.values()],
            "chunks": [chunk.model_dump(mode="json") for chunk in self.chunks.values()],
        }
        self.path.write_text(json.dumps(data, indent=2))

    def upsert_document_with_chunks(self, document: RagDocument, chunks: list[RagChunk]) -> None:
        self.documents[document.id] = document
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        self._save()

    def dense_search(
        self, query_embedding: list[float], top_k: int, filters: RagFilters | None
    ) -> list[RetrievedChunk]:
        results = []
        for chunk in self._filtered_chunks(filters):
            score = _cosine(query_embedding, chunk.embedding or [])
            results.append(self._to_retrieved(chunk, score))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def sparse_search(self, query: str, top_k: int, filters: RagFilters | None) -> list[RetrievedChunk]:
        query_tokens = _token_list(query)
        chunks = self._filtered_chunks(filters)
        if not chunks or not query_tokens:
            return []

        doc_tokens = {chunk.id: _token_list(self._search_text(chunk)) for chunk in chunks}
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            doc_freq.update(set(tokens))
        avg_len = sum(len(tokens) for tokens in doc_tokens.values()) / max(len(doc_tokens), 1)
        total_docs = len(chunks)
        k1 = 1.5
        b = 0.75

        results = []
        for chunk in chunks:
            tokens = doc_tokens[chunk.id]
            counts = Counter(tokens)
            score = 0.0
            doc_len = len(tokens) or 1
            for token in query_tokens:
                if token not in counts:
                    continue
                idf = math.log(1 + (total_docs - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
                tf = counts[token]
                denom = tf + k1 * (1 - b + b * doc_len / max(avg_len, 1))
                score += idf * ((tf * (k1 + 1)) / denom)
            results.append(self._to_retrieved(chunk, score))
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def get_sibling_chunks(self, document_id: str, chunk_index: int, window: int = 1) -> list[RagChunk]:
        siblings = [
            chunk
            for chunk in self.chunks.values()
            if chunk.document_id == document_id and abs(chunk.chunk_index - chunk_index) <= window
        ]
        return sorted(siblings, key=lambda item: item.chunk_index)

    def _filtered_chunks(self, filters: RagFilters | None) -> list[RagChunk]:
        chunks = list(self.chunks.values())
        if filters is None:
            return chunks
        return [chunk for chunk in chunks if self._matches_filters(chunk, filters)]

    def _matches_filters(self, chunk: RagChunk, filters: RagFilters) -> bool:
        if filters.source_type and chunk.source_type != filters.source_type:
            return False
        labels = chunk.metadata.get("labels") or []
        if filters.labels and not set(filters.labels).issubset(set(labels)):
            return False
        if filters.path and chunk.metadata.get("path") != filters.path:
            return False
        if filters.repo and chunk.metadata.get("repo") != filters.repo:
            return False
        created_at = _parse_dt(chunk.metadata.get("created_at"))
        resolved_at = _parse_dt(chunk.metadata.get("resolved_at") or chunk.metadata.get("closed_at"))
        if filters.created_after and created_at and created_at < filters.created_after:
            return False
        if filters.created_before and created_at and created_at > filters.created_before:
            return False
        if filters.resolved_after and resolved_at and resolved_at < filters.resolved_after:
            return False
        if filters.resolved_before and resolved_at and resolved_at > filters.resolved_before:
            return False
        return True

    def _to_retrieved(self, chunk: RagChunk, score: float) -> RetrievedChunk:
        document = self.documents.get(chunk.document_id)
        metadata = dict(chunk.metadata)
        if document is not None:
            metadata.setdefault("parent_id", document.id)
            metadata.setdefault("parent_source_id", document.source_id)
            metadata.setdefault("parent_title", document.title)
            metadata.setdefault("parent_metadata", document.metadata)
            metadata.setdefault("parent_text_preview", document.text[:2000])
        return RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            source_type=chunk.source_type,
            source_id=chunk.source_id,
            title=chunk.title,
            url=chunk.url,
            score=score,
            text=chunk.text,
            metadata=metadata,
        )

    def _search_text(self, chunk: RagChunk) -> str:
        path = chunk.metadata.get("path") or ""
        labels = " ".join(chunk.metadata.get("labels") or [])
        return f"{chunk.title} {chunk.source_id} {path} {labels} {chunk.text}"


class PostgresRagRepository(RagRepository):
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self):
        import psycopg

        return psycopg.connect(self.database_url)

    def upsert_document_with_chunks(self, document: RagDocument, chunks: list[RagChunk]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into rag_documents (id, source_type, source_id, title, url, text, metadata, created_at, updated_at)
                    values (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, now())
                    on conflict (source_type, source_id) do update
                    set title = excluded.title, url = excluded.url, text = excluded.text,
                        metadata = excluded.metadata, updated_at = now()
                    returning id
                    """,
                    (
                        document.id,
                        document.source_type,
                        document.source_id,
                        document.title,
                        document.url,
                        document.text,
                        json.dumps(document.metadata),
                        document.created_at,
                    ),
                )
                document_id = cur.fetchone()[0]
                for chunk in chunks:
                    cur.execute(
                        """
                        insert into rag_chunks (id, document_id, chunk_index, text, embedding, metadata, created_at)
                        values (%s, %s, %s, %s, %s::vector, %s::jsonb, now())
                        on conflict (document_id, chunk_index) do update
                        set text = excluded.text, embedding = excluded.embedding, metadata = excluded.metadata
                        """,
                        (
                            chunk.id,
                            document_id,
                            chunk.chunk_index,
                            chunk.text,
                            _vector_literal(chunk.embedding or []),
                            json.dumps(chunk.metadata),
                        ),
                    )

    def dense_search(
        self, query_embedding: list[float], top_k: int, filters: RagFilters | None
    ) -> list[RetrievedChunk]:
        where, params = self._filter_sql(filters)
        params.extend([_vector_literal(query_embedding), top_k])
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    select c.id, c.document_id, d.source_type, d.source_id, d.title, d.url,
                           1 - (c.embedding <=> %s::vector) as score, c.text, c.metadata,
                           d.metadata, d.text
                    from rag_chunks c
                    join rag_documents d on d.id = c.document_id
                    {where}
                    order by c.embedding <=> %s::vector
                    limit %s
                    """,
                    [params[-2], *params[:-2], params[-2], params[-1]],
                )
                return [_row_to_retrieved(row) for row in cur.fetchall()]

    def sparse_search(self, query: str, top_k: int, filters: RagFilters | None) -> list[RetrievedChunk]:
        where, params = self._filter_sql(filters, prefix="and")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    select c.id, c.document_id, d.source_type, d.source_id, d.title, d.url,
                           ts_rank_cd(
                               to_tsvector('english', d.title || ' ' || d.source_id || ' ' ||
                                   coalesce(d.metadata->>'path', '') || ' ' || c.text),
                               plainto_tsquery('english', %s)
                           ) as score,
                           c.text, c.metadata, d.metadata, d.text
                    from rag_chunks c
                    join rag_documents d on d.id = c.document_id
                    where to_tsvector('english', d.title || ' ' || d.source_id || ' ' ||
                              coalesce(d.metadata->>'path', '') || ' ' || c.text)
                          @@ plainto_tsquery('english', %s)
                    {where}
                    order by score desc
                    limit %s
                    """,
                    [query, query, *params, top_k],
                )
                return [_row_to_retrieved(row) for row in cur.fetchall()]

    def get_sibling_chunks(self, document_id: str, chunk_index: int, window: int = 1) -> list[RagChunk]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select c.id, c.document_id, d.source_type, d.source_id, d.title,
                           c.text, c.metadata, d.url
                    from rag_chunks c
                    join rag_documents d on d.id = c.document_id
                    where c.document_id = %s
                      and abs(c.chunk_index - %s) <= %s
                    order by c.chunk_index
                    """,
                    (document_id, chunk_index, window),
                )
                chunks = []
                for row in cur.fetchall():
                    metadata = row[6] or {}
                    chunks.append(
                        RagChunk(
                            id=str(row[0]),
                            document_id=str(row[1]),
                            source_type=row[2],
                            source_id=row[3],
                            title=row[4],
                            text=row[5],
                            chunk_index=int(metadata.get("chunk_index", 0)),
                            url=row[7],
                            metadata=metadata,
                        )
                    )
                return chunks

    def _filter_sql(self, filters: RagFilters | None, prefix: str = "where") -> tuple[str, list[Any]]:
        clauses = []
        params: list[Any] = []
        if filters:
            if filters.source_type:
                clauses.append("d.source_type = %s")
                params.append(filters.source_type)
            if filters.labels:
                clauses.append("(d.metadata->'labels') ?& %s")
                params.append(filters.labels)
            if filters.path:
                clauses.append("(d.metadata->>'path' = %s or c.metadata->>'path' = %s)")
                params.extend([filters.path, filters.path])
            if filters.repo:
                clauses.append("(d.metadata->>'repo' = %s or c.metadata->>'repo' = %s)")
                params.extend([filters.repo, filters.repo])
            if filters.created_after:
                clauses.append("d.created_at >= %s")
                params.append(filters.created_after)
            if filters.created_before:
                clauses.append("d.created_at <= %s")
                params.append(filters.created_before)
            if filters.resolved_after:
                clauses.append("(d.metadata->>'resolved_at')::timestamptz >= %s")
                params.append(filters.resolved_after)
            if filters.resolved_before:
                clauses.append("(d.metadata->>'resolved_at')::timestamptz <= %s")
                params.append(filters.resolved_before)
        if not clauses:
            return "", []
        return f"{prefix} " + " and ".join(clauses), params


def create_rag_repository(
    database_url: str | None,
    local_store_path: str,
    force_local_store: bool | None = None,
) -> RagRepository:
    if force_local_store is None:
        force_local_store = os.getenv("RAG_FORCE_LOCAL_STORE", "false").lower() == "true"
    if database_url and not force_local_store:
        return PostgresRagRepository(database_url)
    return LocalJsonRagRepository(local_store_path)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _row_to_retrieved(row: tuple[Any, ...]) -> RetrievedChunk:
    metadata = dict(row[8] or {})
    if len(row) > 9:
        parent_metadata = row[9] or {}
        metadata.setdefault("parent_metadata", parent_metadata)
        metadata.setdefault("repo", parent_metadata.get("repo"))
        metadata.setdefault("path", parent_metadata.get("path"))
    if len(row) > 10 and row[10]:
        metadata.setdefault("parent_text_preview", str(row[10])[:2000])
    return RetrievedChunk(
        chunk_id=str(row[0]),
        document_id=str(row[1]),
        source_type=row[2],
        source_id=row[3],
        title=row[4],
        url=row[5],
        score=float(row[6] or 0.0),
        text=row[7],
        metadata=metadata,
    )


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None

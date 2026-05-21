from app.domain.rag import (
    RagFilters,
    RagQueryRequest,
    RagQueryResponse,
    RetrievalDebug,
    RetrievedChunk,
    SourceTypeFilter,
)
from app.infra.embedding_provider import SentenceTransformerEmbeddingProvider
from app.infra.llm_provider import ExtractiveAnswerProvider, LlmProvider
from app.infra.rag_config import RagConfig
from app.infra.reranker import CrossEncoderReranker, Reranker
from app.repositories.rag_repository import RagRepository, create_rag_repository
from app.services.hybrid_retrieval import merge_hybrid_scores
from app.services.query_transformer import QueryTransformer


class RagService:
    def __init__(
        self,
        repository: RagRepository | None = None,
        embedding_provider: SentenceTransformerEmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        llm_provider: LlmProvider | None = None,
        config: RagConfig | None = None,
    ) -> None:
        self.config = config or RagConfig()
        self.repository = repository or create_rag_repository(
            self.config.database_url,
            self.config.local_store_path,
            force_local_store=self.config.force_local_store,
        )
        self.embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider(
            self.config.embedding_model
        )
        self.reranker = reranker or CrossEncoderReranker(
            self.config.reranker_model, enabled=self.config.reranker_enabled
        )
        self.llm_provider = llm_provider or ExtractiveAnswerProvider()
        self.query_transformer = QueryTransformer()

    def query(self, request: RagQueryRequest) -> RagQueryResponse:
        chunks, debug = self.retrieve_advanced(
            request.question,
            top_k=request.top_k,
            source_type=request.source_type,
            filters=request.filters,
        )
        answer = self._answer(request.question, chunks)
        trace = debug.model_dump(mode="json")
        return RagQueryResponse(
            question=request.question,
            answer=answer,
            sources=chunks,
            chunks=chunks,
            retrieval_debug=debug,
            trace=trace,
        )

    def retrieve_naive(
        self,
        question: str,
        top_k: int = 5,
        filters: RagFilters | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_provider.embed([question])[0]
        return self.repository.dense_search(query_embedding, top_k, filters)

    def retrieve_advanced(
        self,
        question: str,
        top_k: int = 5,
        source_type: SourceTypeFilter | None = None,
        filters: RagFilters | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDebug]:
        routed_source_type = source_type or self.route_source_type(question)
        effective_filters = self._apply_source_type_filter(filters, routed_source_type)
        query_variants, _entities = self.query_transformer.generate_query_variants(question)
        candidates_by_id: dict[str, RetrievedChunk] = {}
        score_trace: dict[str, dict[str, float]] = {}
        hybrid_top_k = max(self.config.dense_top_k, self.config.sparse_top_k)

        for query_variant in query_variants:
            hybrid = self._hybrid_retrieve(query_variant, hybrid_top_k, effective_filters)
            for chunk in hybrid:
                existing = candidates_by_id.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    candidates_by_id[chunk.chunk_id] = chunk
                score_trace.setdefault(chunk.chunk_id, {})
                score_trace[chunk.chunk_id][f"hybrid:{query_variant}"] = chunk.score
                score_trace[chunk.chunk_id]["dense_score"] = max(
                    score_trace[chunk.chunk_id].get("dense_score", 0.0), chunk.dense_score
                )
                score_trace[chunk.chunk_id]["sparse_score"] = max(
                    score_trace[chunk.chunk_id].get("sparse_score", 0.0), chunk.sparse_score
                )

        candidates = sorted(candidates_by_id.values(), key=lambda item: item.score, reverse=True)
        rerank_query = query_variants[0] if query_variants else question
        reranked = self.reranker.rerank(rerank_query, candidates)
        final_limit = top_k
        final_chunks = self._with_context(reranked[:final_limit])
        reranker_used = bool(getattr(self.reranker, "last_used", False))
        debug = RetrievalDebug(
            original_query=question,
            rewritten_query=query_variants[0] if query_variants else question,
            routed_source_type=routed_source_type,
            query_variants=query_variants,
            dense_top_k=self.config.dense_top_k,
            sparse_top_k=self.config.sparse_top_k,
            hybrid_top_k=hybrid_top_k,
            reranked_top_k=len(final_chunks),
            hybrid_alpha=self.config.hybrid_alpha,
            reranker_enabled=getattr(self.reranker, "enabled", self.config.reranker_enabled),
            reranker_used=reranker_used,
            reranker_model=getattr(self.reranker, "model_name", None),
            final_chunk_ids=[chunk.chunk_id for chunk in final_chunks],
            reranked_chunk_ids=[chunk.chunk_id for chunk in reranked[:final_limit]],
            scores=score_trace,
        )
        return final_chunks, debug

    def _hybrid_retrieve(
        self,
        query: str,
        limit: int,
        filters: RagFilters | None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_provider.embed([query])[0]
        dense = self.repository.dense_search(query_embedding, self.config.dense_top_k, filters)
        sparse = self.repository.sparse_search(query, self.config.sparse_top_k, filters)
        return merge_hybrid_scores(dense, sparse, self.config.hybrid_alpha, limit)

    def route_source_type(self, question: str) -> SourceTypeFilter:
        lowered = question.lower()
        doc_keywords = ("documentation", "docs", "guide", "readme", "api usage", "manual")
        issue_keywords = (
            "similar issue",
            "resolved issue",
            "maintainer answer",
            "bug history",
            "previous report",
            "fixed before",
        )
        if any(keyword in lowered for keyword in doc_keywords):
            return "doc"
        if any(keyword in lowered for keyword in issue_keywords):
            return "issue"
        return "all"

    def _apply_source_type_filter(
        self, filters: RagFilters | None, source_type: SourceTypeFilter
    ) -> RagFilters | None:
        if source_type == "all":
            return filters
        if filters is None:
            return RagFilters(source_type=source_type)
        return filters.model_copy(update={"source_type": source_type})

    def _with_context(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        enriched = []
        for chunk in chunks:
            siblings = self.repository.get_sibling_chunks(
                chunk.document_id, int(chunk.metadata.get("chunk_index", 0)), window=1
            )
            sibling_text = "\n\n".join(
                sibling.text for sibling in siblings if sibling.id != chunk.chunk_id
            )
            metadata = dict(chunk.metadata)
            if sibling_text:
                metadata["sibling_context"] = sibling_text[:1200]
            enriched.append(chunk.model_copy(update={"metadata": metadata}))
        return enriched

    def _answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks or chunks[0].score < self.config.min_answer_score:
            return "I could not find enough evidence in the docs or resolved issues."
        answer = self.llm_provider.generate_answer(question, chunks)
        return answer or "I could not find enough evidence in the docs or resolved issues."

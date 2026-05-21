import os


class RagConfig:
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model: str = os.getenv(
        "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker_enabled: bool = os.getenv("RAG_RERANKER_ENABLED", "true").lower() == "true"
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    dense_top_k: int = int(os.getenv("RAG_DENSE_TOP_K", "20"))
    sparse_top_k: int = int(os.getenv("RAG_SPARSE_TOP_K", "20"))
    final_top_k: int = int(os.getenv("RAG_FINAL_TOP_K", "8"))
    hybrid_alpha: float = float(os.getenv("RAG_HYBRID_ALPHA", "0.6"))
    database_url: str | None = os.getenv("DATABASE_URL")
    local_store_path: str = os.getenv("RAG_LOCAL_STORE_PATH", "data/rag_store.json")
    force_local_store: bool = os.getenv("RAG_FORCE_LOCAL_STORE", "false").lower() == "true"
    min_answer_score: float = float(os.getenv("RAG_MIN_ANSWER_SCORE", "0.05"))

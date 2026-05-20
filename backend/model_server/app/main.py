from fastapi import FastAPI

from app.api.classification import router as classification_router
from app.api.ner import router as ner_router
from app.api.rag import router as rag_router
from app.api.summarization import router as summarization_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Maintainer's Copilot Model Server",
        version="0.1.0-week7",
        description="Model server for classification, NER, summarization, and RAG.",
    )

    app.include_router(classification_router)
    app.include_router(ner_router)
    app.include_router(summarization_router)
    app.include_router(rag_router)

    @app.get("/health/live")
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

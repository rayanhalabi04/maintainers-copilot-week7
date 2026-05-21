from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.api.classification import router as classification_router
from app.api.memory import router as memory_router
from app.api.ner import router as ner_router
from app.api.rag import router as rag_router
from app.api.summarization import router as summarization_router
from app.api.widget import router as widget_router
from app.services.startup_validation import validate_startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_startup()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Maintainer's Copilot Model Server",
        version="0.1.0-week7",
        description="Model server for classification, NER, summarization, and RAG.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(classification_router)
    app.include_router(ner_router)
    app.include_router(summarization_router)
    app.include_router(rag_router)
    app.include_router(chat_router)
    app.include_router(auth_router)
    app.include_router(memory_router)
    app.include_router(widget_router)

    @app.get("/health/live")
    def health_live() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

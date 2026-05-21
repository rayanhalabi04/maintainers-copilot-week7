import os
from pathlib import Path

from app.infra.artifact_validation import validate_classifier_artifacts
from app.infra.eval_thresholds import load_required_thresholds
from app.infra.vault_client import vault_available


class StartupValidationError(RuntimeError):
    pass


def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "eval_thresholds.yaml").exists():
            return parent
        if (parent / "artifacts").exists():
            return parent

    # Docker fallback: /app/app/services/startup_validation.py -> /app
    return Path("/app")


ROOT = find_project_root()
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def strict_startup_enabled() -> bool:
    return (
        os.getenv("APP_ENV", "").lower() == "production"
        or os.getenv("STRICT_STARTUP_VALIDATION", "").lower() == "true"
    )


def validate_startup() -> None:
    strict = strict_startup_enabled()
    threshold_path = Path(os.getenv("EVAL_THRESHOLDS_PATH", str(ROOT / "eval_thresholds.yaml")))
    load_required_thresholds(threshold_path)
    validate_classifier_artifacts(ARTIFACT_DIR)

    if strict and not _local_dev_mode_enabled(strict):
        if not vault_available(timeout_seconds=2):
            raise StartupValidationError(
                "Vault is unreachable. Set LOCAL_DEV_MODE=true only for local development."
            )

    database_url = os.getenv("DATABASE_URL")
    force_local_store = os.getenv("RAG_FORCE_LOCAL_STORE", "false").lower() == "true"
    if database_url and not force_local_store:
        _validate_rag_database(database_url)
    elif strict and force_local_store:
        raise StartupValidationError("Production RAG cannot boot with RAG_FORCE_LOCAL_STORE=true.")

    llm_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if strict and llm_provider == "groq" and not os.getenv("GROQ_API_KEY", "").strip():
        raise StartupValidationError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")


def _local_dev_mode_enabled(strict: bool) -> bool:
    if os.getenv("LOCAL_DEV_MODE", "false").lower() == "true":
        return True
    if strict:
        return False
    return os.getenv("APP_ENV", "").lower() in {
        "",
        "local",
        "development",
        "dev",
        "test",
    }


def _validate_rag_database(database_url: str) -> None:
    try:
        import psycopg
    except Exception as exc:
        raise StartupValidationError("psycopg is required for production RAG database validation.") from exc

    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("select to_regclass('public.rag_documents'), to_regclass('public.rag_chunks')")
                rag_documents, rag_chunks = cur.fetchone()
    except Exception as exc:
        raise StartupValidationError(f"RAG database is not reachable or not usable: {exc}") from exc

    missing = []
    if rag_documents is None:
        missing.append("rag_documents")
    if rag_chunks is None:
        missing.append("rag_chunks")
    if missing:
        raise StartupValidationError(f"Required RAG tables are missing: {', '.join(missing)}")

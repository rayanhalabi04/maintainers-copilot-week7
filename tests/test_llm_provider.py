from app.infra.llm_provider import (
    DEFAULT_GROQ_MODEL,
    GroqToolCallingProvider,
    get_tool_calling_llm_provider,
)


def test_groq_provider_selected_when_configured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    provider = get_tool_calling_llm_provider()

    assert isinstance(provider, GroqToolCallingProvider)
    assert provider.model == DEFAULT_GROQ_MODEL


def test_groq_provider_uses_configured_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "custom-model")

    provider = get_tool_calling_llm_provider()

    assert isinstance(provider, GroqToolCallingProvider)
    assert provider.model == "custom-model"


def test_groq_is_selected_when_key_exists_without_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    provider = get_tool_calling_llm_provider()

    assert isinstance(provider, GroqToolCallingProvider)


def test_no_provider_without_key_in_non_strict_mode(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("STRICT_STARTUP_VALIDATION", raising=False)

    assert get_tool_calling_llm_provider() is None

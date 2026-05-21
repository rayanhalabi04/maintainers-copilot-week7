import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import requests

from app.domain.rag import RetrievedChunk


LLM_PROVIDER_ENV = "LLM_PROVIDER"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MODEL_ENV = "GROQ_MODEL"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


class LlmProvider:
    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str | None:
        return None


class ExtractiveAnswerProvider(LlmProvider):
    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str | None:
        if not chunks:
            return None
        lead = chunks[0]
        prefix = "Based on the retrieved project evidence"
        if lead.source_type == "issue":
            prefix = "Based on previous resolved issues"
        evidence = lead.text
        sibling_context = lead.metadata.get("sibling_context")
        if sibling_context and len(evidence.split()) < 12:
            evidence = f"{evidence}. {sibling_context}"
        text = evidence.strip().replace("\n", " ")
        if len(text) > 650:
            text = f"{text[:647]}..."
        return f"{prefix} from {lead.title}, the most relevant evidence says: {text}"


LlmRole = Literal["system", "user", "assistant", "tool"]


@dataclass
class LlmMessage:
    role: LlmRole
    content: str | None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


@dataclass
class LlmToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class LlmToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LlmChatResponse:
    content: str | None
    tool_calls: list[LlmToolCall]
    raw: dict[str, Any] | None = None


class ToolCallingLlmProvider:
    provider_name = "base"

    def chat(
        self,
        messages: list[LlmMessage],
        tools: list[LlmToolSpec],
    ) -> LlmChatResponse:
        raise NotImplementedError


class GroqToolCallingProvider(ToolCallingLlmProvider):
    provider_name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_seconds: int = 30,
    ) -> None:
        if not api_key:
            raise ValueError("Groq API key is required.")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def chat(
        self,
        messages: list[LlmMessage],
        tools: list[LlmToolSpec],
    ) -> LlmChatResponse:
        payload = {
            "model": self.model,
            "messages": [self._message_payload(message) for message in messages],
            "tools": [self._tool_payload(tool) for tool in tools],
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        return LlmChatResponse(
            content=message.get("content"),
            tool_calls=self._parse_tool_calls(message.get("tool_calls") or []),
            raw={
                "id": data.get("id"),
                "model": data.get("model"),
                "usage": data.get("usage"),
            },
        )

    def _message_payload(self, message: LlmMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content or "",
        }
        if message.role == "tool" and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        return payload

    def _tool_payload(self, tool: LlmToolSpec) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _parse_tool_calls(self, raw_tool_calls: list[dict[str, Any]]) -> list[LlmToolCall]:
        parsed: list[LlmToolCall] = []
        for index, raw_call in enumerate(raw_tool_calls):
            function = raw_call.get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                decoded_args = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                decoded_args = {}
            parsed.append(
                LlmToolCall(
                    id=raw_call.get("id") or f"tool-call-{index}",
                    name=function.get("name") or "",
                    arguments=decoded_args if isinstance(decoded_args, dict) else {},
                )
            )
        return parsed


def strict_llm_required() -> bool:
    return (
        os.getenv("APP_ENV", "").lower() == "production"
        or os.getenv("STRICT_STARTUP_VALIDATION", "").lower() == "true"
    )


def get_tool_calling_llm_provider() -> ToolCallingLlmProvider | None:
    api_key = os.getenv(GROQ_API_KEY_ENV, "").strip()
    provider = os.getenv(LLM_PROVIDER_ENV, "").strip().lower()
    if not provider and api_key:
        provider = "groq"
    if not provider:
        return None

    if provider != "groq":
        if strict_llm_required():
            raise RuntimeError(f"Unsupported LLM_PROVIDER={provider!r}.")
        return None

    if not api_key:
        if strict_llm_required():
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq.")
        return None

    model = os.getenv(GROQ_MODEL_ENV, DEFAULT_GROQ_MODEL).strip() or DEFAULT_GROQ_MODEL
    return GroqToolCallingProvider(api_key=api_key, model=model)

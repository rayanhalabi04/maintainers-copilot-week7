import json
import re
from pathlib import Path
from typing import Any

from app.domain.auth import CurrentUser
from app.domain.chat import ChatRequest, ChatResponse, ChatToolCall
from app.domain.classification import ClassifyResponse
from app.domain.ner import NerResponse
from app.domain.rag import RagQueryResponse
from app.domain.summarization import SummarizeResponse
from app.infra.llm_provider import (
    LlmChatResponse,
    LlmMessage,
    LlmToolCall,
    LlmToolSpec,
    ToolCallingLlmProvider,
    get_tool_calling_llm_provider,
)
from app.infra.redaction import redact_obj, redact_text
from app.services.chat_tools import ChatTools, ToolResult


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_PROMPT_PATH = PROMPT_DIR / "chat_system_prompt.txt"
TOOL_POLICY_PATH = PROMPT_DIR / "tool_policy.txt"
XML_TOOL_CALL_RE = re.compile(
    r"<(?P<name>[A-Za-z_][A-Za-z0-9_]*)>\s*(?P<args>.*?)\s*</(?P<close>[A-Za-z_][A-Za-z0-9_]*)>",
    re.DOTALL,
)


class ChatService:
    """Single-LLM tool-calling chat service with deterministic local fallback."""

    _ISSUE_KEYWORDS = (
        "bug",
        "error",
        "crash",
        "fails",
        "failure",
        "feature",
        "docs",
        "question",
        "regression",
        "issue",
    )
    _RAG_KEYWORDS = (
        "explain",
        "similar",
        "similar issue",
        "resolved",
        "docs",
        "documentation",
        "guidance",
        "guide",
        "root cause",
        "what should",
        "what to do",
        "next step",
        "maintainer",
        "triage",
        "fix",
        "fixed",
    )
    _ISSUE_CONTEXT_RAG_KEYWORDS = (
        "guidance",
        "guide",
        "maintainer",
        "triage",
        "what should",
        "what to do",
        "next step",
        "next",
        "action",
        "recommend",
        "suggest",
    )
    _CODE_PATTERNS = (
        r"`[^`]+`",
        r"\b[A-Z][A-Za-z]+Error\b",
        r"\b[A-Z_]{3,}\b",
        r"\b[\w./-]+\.(js|ts|json|md|py|yml|yaml|txt|c|cc|h)\b",
        r"\bv?\d+\.\d+(?:\.\d+)?\b",
        r"\b(?:npm|node|npx|yarn|pnpm|python|pytest)\s+[\w:./-]+",
        r"\b(?:at|File) .+:\d+",
        r"https?://[^\s]+",
        r"#\d+",
    )

    def __init__(
        self,
        tools: ChatTools | None = None,
        llm_provider: ToolCallingLlmProvider | None = None,
        enable_llm: bool | None = None,
    ) -> None:
        self.tools = tools or ChatTools()
        self.llm_provider = llm_provider if enable_llm is not False else None
        if enable_llm is not False and llm_provider is None:
            self.llm_provider = get_tool_calling_llm_provider()

    def chat(
        self,
        request: ChatRequest,
        current_user: CurrentUser | None = None,
    ) -> ChatResponse:
        if self.llm_provider is not None:
            return self._chat_with_llm(request, current_user)
        return self._chat_with_fallback(request)

    def _chat_with_fallback(self, request: ChatRequest) -> ChatResponse:
        combined_text = self._build_combined_text(request)
        tool_results: list[ToolResult] = []

        should_classify = bool(request.issue_title or request.issue_body) or self._looks_like_issue(
            request.message
        )
        should_extract = self._has_code_like_patterns(combined_text)
        should_summarize = len(self._body_or_message(request)) > 1000
        should_rag = request.use_rag and self._asks_for_rag(request)

        if should_classify:
            tool_results.append(
                self.tools.classify_issue_tool(
                    text=combined_text,
                    title=request.issue_title,
                    body=request.issue_body or request.message,
                )
            )
        if should_extract:
            tool_results.append(self.tools.extract_entities_tool(combined_text))
        if should_summarize:
            tool_results.append(self.tools.summarize_issue_tool(self._body_or_message(request)))
        if should_rag:
            tool_results.append(self.tools.rag_search_tool(request.message, request.top_k))

        answer = self._build_answer(tool_results)
        trace = {
            "deterministic_router": True,
            "memory_writes": "explicit_only",
            "called_tools": [result.tool_name for result in tool_results],
            "routing": {
                "classify_issue": should_classify,
                "extract_entities": should_extract,
                "summarize_thread": should_summarize,
                "rag_query": should_rag,
            },
        }
        return ChatResponse(
            answer=answer,
            tool_calls=[
                ChatToolCall(
                    tool_name=result.tool_name,
                    status=result.status,  # type: ignore[arg-type]
                    summary=redact_text(result.summary) if result.summary else None,
                    error=redact_text(result.error) if result.error else None,
                )
                for result in tool_results
            ],
            trace=redact_obj(trace),
        )

    def _chat_with_llm(
        self,
        request: ChatRequest,
        current_user: CurrentUser | None,
    ) -> ChatResponse:
        assert self.llm_provider is not None
        messages = [
            LlmMessage("system", self._load_system_prompt()),
            LlmMessage("user", self._build_user_prompt(request)),
        ]
        tool_results: list[ToolResult] = []
        assistant_tool_calls: list[dict[str, Any]] = []

        first_response = self.llm_provider.chat(messages, self.tool_specs())
        structured_tool_calls = first_response.tool_calls
        xml_tool_calls = []
        if not structured_tool_calls:
            xml_tool_calls = self._parse_xml_tool_calls(first_response.content or "")

        for tool_call in [*structured_tool_calls, *xml_tool_calls]:
            result = self._execute_llm_tool_call(tool_call, request, current_user)
            tool_results.append(result)

        if tool_results:
            if structured_tool_calls:
                for tool_call, result in zip(structured_tool_calls, tool_results, strict=False):
                    assistant_tool_calls.append(self._assistant_tool_call_payload(tool_call))
                    messages.append(
                        LlmMessage(
                            "tool",
                            self._tool_result_content(result),
                            tool_call_id=tool_call.id,
                        )
                    )
                messages.insert(
                    2,
                    LlmMessage(
                        "assistant",
                        first_response.content,
                        tool_calls=assistant_tool_calls,
                    ),
                )
            else:
                messages.append(
                    LlmMessage(
                        "assistant",
                        self._strip_xml_tool_calls(first_response.content or "").strip()
                        or "I will use the selected tools.",
                    )
                )
                messages.append(
                    LlmMessage(
                        "user",
                        (
                            "Tool results are below. Write the final maintainer-facing answer. "
                            "Do not include XML tool tags or raw tool-call markup.\n\n"
                            f"{self._tool_results_content(tool_results)}"
                        ),
                    )
                )
            final_response = self.llm_provider.chat(messages, self.tool_specs())
        else:
            final_response = first_response

        answer = final_response.content or self._build_answer(tool_results)
        if self._contains_xml_tool_call(answer):
            answer = self._build_answer(tool_results)
        trace = {
            "deterministic_router": False,
            "llm_provider": self.llm_provider.provider_name,
            "memory_writes": "explicit_only",
            "called_tools": [result.tool_name for result in tool_results],
            "llm": self._redacted_llm_trace(final_response),
        }
        return ChatResponse(
            answer=redact_text(answer),
            tool_calls=[
                ChatToolCall(
                    tool_name=result.tool_name,
                    status=result.status,  # type: ignore[arg-type]
                    summary=redact_text(result.summary) if result.summary else None,
                    error=redact_text(result.error) if result.error else None,
                )
                for result in tool_results
            ],
            trace=redact_obj(trace),
        )

    def _build_combined_text(self, request: ChatRequest) -> str:
        parts = [
            f"Title: {request.issue_title}" if request.issue_title else "",
            f"Body: {request.issue_body}" if request.issue_body else "",
            f"Message: {request.message}",
        ]
        return "\n\n".join(part for part in parts if part).strip()

    def _build_user_prompt(self, request: ChatRequest) -> str:
        return (
            f"{self._build_combined_text(request)}\n\n"
            f"RAG enabled: {request.use_rag}\n"
            f"RAG top_k: {request.top_k}"
        ).strip()

    def _load_system_prompt(self) -> str:
        prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        tool_policy = TOOL_POLICY_PATH.read_text(encoding="utf-8")
        return f"{prompt}\n\n{tool_policy}".strip()

    def _body_or_message(self, request: ChatRequest) -> str:
        return request.issue_body or request.message

    def _looks_like_issue(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in self._ISSUE_KEYWORDS)

    def _has_code_like_patterns(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self._CODE_PATTERNS)

    def _asks_for_rag(self, request: ChatRequest) -> bool:
        lowered_message = request.message.lower()
        if any(keyword in lowered_message for keyword in self._RAG_KEYWORDS):
            return True

        has_issue_context = bool(request.issue_title or request.issue_body)
        asks_for_action = any(
            keyword in lowered_message for keyword in self._ISSUE_CONTEXT_RAG_KEYWORDS
        )
        return has_issue_context and asks_for_action

    def _build_answer(self, tool_results: list[ToolResult]) -> str:
        if not tool_results:
            return (
                "I did not need a tool for this short request.\n\n"
                "Suggested maintainer action: ask for an issue title/body or a specific "
                "triage question so I can classify, extract entities, summarize, or search evidence."
            )

        sections: list[str] = []
        classification = self._first_ok(tool_results, "classify_issue")
        entities = self._first_ok(tool_results, "extract_entities")
        summary = self._first_ok(tool_results, "summarize_thread")
        rag = self._first_ok(tool_results, "rag_query")

        if classification and isinstance(classification.data, ClassifyResponse):
            sections.append(
                f"Likely label: {classification.data.label} "
                f"(confidence {classification.data.confidence:.2f})."
            )
        if entities and isinstance(entities.data, NerResponse):
            entity_text = self._format_entities(entities.data)
            if entity_text:
                sections.append(f"Important entities: {redact_text(entity_text)}.")
        if summary and isinstance(summary.data, SummarizeResponse):
            sections.append(f"Short summary: {redact_text(summary.data.summary)}")
        if rag and isinstance(rag.data, RagQueryResponse):
            sections.append(self._format_rag_guidance(rag.data))

        failed = [result for result in tool_results if result.status == "error"]
        if failed:
            failed_names = ", ".join(result.tool_name for result in failed)
            sections.append(f"Some tool evidence is unavailable: {failed_names}.")

        sections.append(f"Suggested maintainer action: {self._suggest_action(tool_results)}")
        return redact_text("\n\n".join(section for section in sections if section))

    def _first_ok(self, tool_results: list[ToolResult], tool_name: str) -> ToolResult | None:
        return next(
            (
                result
                for result in tool_results
                if result.tool_name == tool_name and result.status == "ok"
            ),
            None,
        )

    def _format_entities(self, response: NerResponse) -> str:
        entities = []
        seen: set[tuple[str, str]] = set()
        for entity in response.entities:
            key = (entity.label, entity.text)
            if key in seen:
                continue
            seen.add(key)
            entities.append(f"{redact_text(entity.text)} ({entity.label})")
            if len(entities) >= 8:
                break
        return ", ".join(entities)

    def _format_rag_guidance(self, response: RagQueryResponse) -> str:
        if not response.sources:
            return "Retrieved evidence: no strong matching docs or resolved issues were found."

        source_bits = []
        for source in response.sources[:3]:
            source_bits.append(
                f"{redact_text(source.title)} [{source.source_type}, score {source.score:.2f}]"
            )
        return (
            f"Retrieved evidence / guidance: {redact_text(response.answer)}\n"
            f"Top sources: {', '.join(source_bits)}"
        )

    def _suggest_action(self, tool_results: list[ToolResult]) -> str:
        classification = self._first_ok(tool_results, "classify_issue")
        rag = self._first_ok(tool_results, "rag_query")
        entities = self._first_ok(tool_results, "extract_entities")

        label = None
        if classification and isinstance(classification.data, ClassifyResponse):
            label = classification.data.label

        if rag and isinstance(rag.data, RagQueryResponse) and rag.data.sources:
            return "compare the report with the retrieved evidence, then ask for a minimal reproduction if the fix is still unclear."
        if label == "bug":
            return "label as bug and request a minimal reproduction, affected version, and expected versus actual behavior."
        if label == "docs":
            return "label as docs and point the reporter to the relevant documentation area or ask what wording is confusing."
        if label == "feature":
            return "label as feature and ask for motivation, user impact, and a concrete API or behavior proposal."
        if entities:
            return "verify the extracted files/errors/versions and ask for missing environment details."
        return "review the issue manually and ask for reproduction details if the report is incomplete."

    def _result_payload(self, result: ToolResult) -> dict[str, Any] | None:
        data = result.data
        if hasattr(data, "model_dump"):
            return data.model_dump(mode="json")
        return None

    def tool_specs(self) -> list[LlmToolSpec]:
        return [
            LlmToolSpec(
                name="classify_issue",
                description="Classify a GitHub issue as bug, feature, docs, or question.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "title": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["text"],
                },
            ),
            LlmToolSpec(
                name="extract_entities",
                description="Extract files, commands, versions, errors, stack traces, URLs, and issue references.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            LlmToolSpec(
                name="summarize_thread",
                description="Summarize a long issue thread or user-provided issue text.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            LlmToolSpec(
                name="rag_query",
                description="Retrieve project documentation or similar resolved issues and produce grounded evidence.",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                    "required": ["question"],
                },
            ),
            LlmToolSpec(
                name="write_memory",
                description="Explicitly store a memory only when the user asks to remember, save, or store it.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["episodic", "semantic", "procedural"],
                        },
                        "metadata": {"type": "object"},
                    },
                    "required": ["text"],
                },
            ),
        ]

    def _execute_llm_tool_call(
        self,
        tool_call: LlmToolCall,
        request: ChatRequest,
        current_user: CurrentUser | None,
    ) -> ToolResult:
        args = tool_call.arguments
        tool_error = args.get("_tool_error")
        if tool_error:
            return ToolResult(
                tool_call.name or "unknown_tool",
                "error",
                error=str(tool_error),
            )
        combined_text = self._build_combined_text(request)
        if tool_call.name == "classify_issue":
            return self.tools.classify_issue_tool(
                text=str(args.get("text") or combined_text),
                title=args.get("title") or request.issue_title,
                body=args.get("body") or request.issue_body or request.message,
            )
        if tool_call.name == "extract_entities":
            return self.tools.extract_entities_tool(str(args.get("text") or combined_text))
        if tool_call.name == "summarize_thread":
            return self.tools.summarize_issue_tool(str(args.get("text") or self._body_or_message(request)))
        if tool_call.name == "rag_query":
            if not request.use_rag:
                return ToolResult("rag_query", "skipped", summary="RAG is disabled for this request.")
            return self.tools.rag_search_tool(
                question=str(args.get("question") or request.message),
                top_k=int(args.get("top_k") or request.top_k),
            )
        if tool_call.name == "write_memory":
            if current_user is None:
                return ToolResult("write_memory", "error", error="Memory write requires an authenticated user.")
            return self.tools.write_memory_tool(
                current_user=current_user,
                text=str(args.get("text") or ""),
                memory_type=str(args.get("memory_type") or "episodic"),
                metadata=args.get("metadata") if isinstance(args.get("metadata"), dict) else {},
            )
        return ToolResult(tool_call.name or "unknown_tool", "error", error="Unknown chat tool.")

    def _parse_xml_tool_calls(self, content: str) -> list[LlmToolCall]:
        tool_calls: list[LlmToolCall] = []
        allowed_tools = {tool.name for tool in self.tool_specs()}
        for index, match in enumerate(XML_TOOL_CALL_RE.finditer(content)):
            tool_name = match.group("name")
            closing_tag = match.group("close")
            raw_args = match.group("args")
            if closing_tag not in {tool_name, "function"}:
                continue
            if tool_name not in allowed_tools:
                tool_calls.append(
                    LlmToolCall(
                        id=f"xml-tool-call-{index}",
                        name=tool_name,
                        arguments={
                            "_tool_error": "Unknown chat tool.",
                            "_raw_arguments": raw_args,
                        },
                    )
                )
                continue
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {
                    "_tool_error": "Invalid tool JSON.",
                    "_raw_arguments": raw_args,
                }
            if not isinstance(parsed_args, dict):
                parsed_args = {
                    "_tool_error": "Tool JSON must be an object.",
                    "_raw_arguments": raw_args,
                }
            tool_calls.append(
                LlmToolCall(
                    id=f"xml-tool-call-{index}",
                    name=tool_name,
                    arguments=parsed_args,
                )
            )
        return tool_calls

    def _contains_xml_tool_call(self, content: str) -> bool:
        return bool(XML_TOOL_CALL_RE.search(content))

    def _strip_xml_tool_calls(self, content: str) -> str:
        return XML_TOOL_CALL_RE.sub("", content)

    def _assistant_tool_call_payload(self, tool_call: LlmToolCall) -> dict[str, Any]:
        return {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": redact_text(json.dumps(tool_call.arguments)),
            },
        }

    def _tool_result_content(self, result: ToolResult) -> str:
        return redact_text(
            str(
                {
                    "tool_name": result.tool_name,
                    "status": result.status,
                    "summary": result.summary,
                    "error": result.error,
                    "data": self._result_payload(result),
                }
            )
        )

    def _tool_results_content(self, tool_results: list[ToolResult]) -> str:
        return "\n".join(self._tool_result_content(result) for result in tool_results)

    def _redacted_llm_trace(self, response: LlmChatResponse) -> dict[str, Any]:
        return redact_obj(
            {
                "tool_call_count": len(response.tool_calls),
                "raw": response.raw,
            }
        )

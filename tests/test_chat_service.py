from app.domain.chat import ChatRequest
from app.domain.auth import CurrentUser
from app.domain.classification import ClassifyResponse
from app.infra.llm_provider import LlmChatResponse, LlmMessage, LlmToolCall, ToolCallingLlmProvider
from app.domain.rag import RagQueryResponse, RagSource, RetrievalDebug
from app.services.chat_service import ChatService
from app.services.chat_tools import ToolResult


class FakeChatTools:
    def __init__(self) -> None:
        self.called: list[str] = []
        self.memory_writes: list[str] = []

    def classify_issue_tool(self, text, title=None, body=None):
        self.called.append("classify_issue")
        return ToolResult(
            tool_name="classify_issue",
            status="ok",
            summary="bug (0.91)",
            data=ClassifyResponse(
                label="bug",
                confidence=0.91,
                probabilities={"bug": 0.91, "feature": 0.03, "docs": 0.03, "question": 0.03},
                model_name="fake",
            ),
        )

    def extract_entities_tool(self, text):
        self.called.append("extract_entities")
        return ToolResult(
            tool_name="extract_entities",
            status="ok",
            summary="No code-shaped entities found.",
        )

    def summarize_issue_tool(self, text):
        self.called.append("summarize_thread")
        return ToolResult(
            tool_name="summarize_thread",
            status="ok",
            summary="Short summary.",
        )

    def rag_search_tool(self, question, top_k=5):
        self.called.append("rag_search")
        return ToolResult(
            tool_name="rag_query",
            status="ok",
            summary="1 sources. Check similar issue.",
            data=RagQueryResponse(
                question=question,
                answer="Check the similar resolved issue before closing.",
                sources=[
                    RagSource(
                        chunk_id="chunk-1",
                        document_id="doc-1",
                        source_type="issue",
                        source_id="101",
                        title="Similar failure",
                        score=0.88,
                        text="Resolved by updating the failing assertion.",
                    )
                ],
                retrieval_debug=RetrievalDebug(
                    original_query=question,
                    rewritten_query=question,
                    dense_top_k=20,
                    sparse_top_k=20,
                    hybrid_top_k=20,
                    reranked_top_k=1,
                    hybrid_alpha=0.6,
                    reranker_enabled=False,
                ),
            ),
        )

    def write_memory_tool(self, current_user, text, memory_type="episodic", metadata=None):
        self.called.append("write_memory")
        self.memory_writes.append(text)
        return ToolResult(
            tool_name="write_memory",
            status="ok",
            summary=f"Stored {memory_type} memory.",
        )


class FakeLlmProvider(ToolCallingLlmProvider):
    provider_name = "fake-llm"

    def __init__(
        self,
        first_tool_calls: list[LlmToolCall] | None = None,
        first_content: str = "I will inspect the issue.",
        final_content: str = "Grounded answer from the selected tools.",
    ) -> None:
        self.calls: list[tuple[list[LlmMessage], list[str]]] = []
        self.first_tool_calls = first_tool_calls or []
        self.first_content = first_content
        self.final_content = final_content

    def chat(self, messages, tools):
        self.calls.append((messages, [tool.name for tool in tools]))
        if len(self.calls) == 1:
            return LlmChatResponse(
                content=self.first_content,
                tool_calls=self.first_tool_calls,
                raw={"model": "fake"},
            )
        return LlmChatResponse(
            content=self.final_content,
            tool_calls=[],
            raw={"model": "fake"},
        )


def test_chat_service_calls_classifier_for_issue_context():
    tools = FakeChatTools()
    service = ChatService(tools=tools)

    response = service.chat(
        ChatRequest(
            message="Please triage this issue.",
            issue_title="fs test fails on Windows",
            issue_body="The test suite fails with an assertion error.",
            use_rag=False,
        )
    )

    assert "classify_issue" in tools.called
    assert response.tool_calls[0].tool_name == "classify_issue"
    assert "Likely label: bug" in response.answer


def test_chat_service_calls_rag_for_maintainer_guidance():
    tools = FakeChatTools()
    service = ChatService(tools=tools)

    response = service.chat(
        ChatRequest(
            message="What should the maintainer do next? Find similar resolved issues.",
            issue_title="AssertionError in timers test",
            issue_body="Timers test fails intermittently.",
            use_rag=True,
            top_k=3,
        )
    )

    assert "rag_search" in tools.called
    assert any(call.tool_name == "rag_query" for call in response.tool_calls)
    assert "Retrieved evidence / guidance" in response.answer


def test_chat_service_calls_rag_for_triage_with_issue_context():
    tools = FakeChatTools()
    service = ChatService(tools=tools)

    response = service.chat(
        ChatRequest(
            message="Can you triage this issue?",
            issue_title="Build fails on macOS with node-gyp error",
            issue_body="When I run npm install, node-gyp fails with a Python version error.",
            use_rag=True,
            top_k=5,
        )
    )

    assert "rag_search" in tools.called
    assert response.trace["routing"]["rag_query"] is True
    assert any(call.tool_name == "rag_query" for call in response.tool_calls)
    assert "Retrieved evidence / guidance" in response.answer


def test_fallback_works_without_groq_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    tools = FakeChatTools()
    service = ChatService(tools=tools)

    response = service.chat(
        ChatRequest(
            message="Please triage this bug.",
            issue_title="fs test fails",
            issue_body="The test fails with ERR_ASSERTION.",
            use_rag=False,
        )
    )

    assert response.trace["deterministic_router"] is True
    assert "classify_issue" in tools.called
    assert response.answer


def test_llm_tools_are_exposed_registered():
    service = ChatService(tools=FakeChatTools(), enable_llm=False)

    tool_names = {tool.name for tool in service.tool_specs()}

    assert tool_names == {
        "classify_issue",
        "extract_entities",
        "summarize_thread",
        "rag_query",
        "write_memory",
    }


def test_llm_provider_uses_single_tool_calling_path():
    tools = FakeChatTools()
    llm = FakeLlmProvider(
        [
            LlmToolCall(
                id="call-1",
                name="classify_issue",
                arguments={"text": "Bug report: fs test fails"},
            ),
            LlmToolCall(
                id="call-2",
                name="rag_query",
                arguments={"question": "Find similar resolved issues", "top_k": 3},
            ),
        ]
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(
        ChatRequest(
            message="What should I do with this issue?",
            issue_title="fs test fails",
            issue_body="ERR_ASSERTION in test-fs.js",
            use_rag=True,
            top_k=3,
        )
    )

    assert response.trace["deterministic_router"] is False
    assert response.trace["llm_provider"] == "fake-llm"
    assert [call.tool_name for call in response.tool_calls] == ["classify_issue", "rag_query"]
    assert len(llm.calls) == 2


def test_xml_tool_call_output_is_parsed_and_tools_execute():
    tools = FakeChatTools()
    llm = FakeLlmProvider(
        first_content=(
            '<rag_query>{"question":"How to handle a malformed URL?","top_k":5}</rag_query>\n'
            '<extract_entities>{"text":"Security concern in URL parsing at lib/url.js"}</extract_entities>\n'
            '<classify_issue>{"text":"Security concern in URL parsing"}</classify_issue>'
        ),
        final_content="Likely label: bug. Retrieved RAG guidance: check validation paths.",
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(
        ChatRequest(
            message="How should we triage this malformed URL validation issue?",
            issue_title="Malformed URL may bypass validation",
            issue_body="Security concern in URL parsing at lib/url.js",
            use_rag=True,
        )
    )

    assert response.trace["deterministic_router"] is False
    assert response.trace["llm_provider"] == "fake-llm"
    assert [call.tool_name for call in response.tool_calls] == [
        "rag_query",
        "extract_entities",
        "classify_issue",
    ]
    assert "rag_search" in tools.called
    assert "extract_entities" in tools.called
    assert "classify_issue" in tools.called
    assert "<rag_query>" not in response.answer
    assert "<classify_issue>" not in response.answer


def test_malformed_function_closed_xml_tool_call_is_parsed_and_executed():
    tools = FakeChatTools()
    malformed_markup = (
        '<rag_query>{"question":"How to handle a security concern in URL parsing?",'
        '"top_k":5}</function>'
    )
    llm = FakeLlmProvider(
        first_content=malformed_markup,
        final_content="Use retrieved guidance to compare URL validation behavior.",
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(
        ChatRequest(
            message="How should we handle a security concern in URL parsing?",
            issue_title="Security concern in URL parsing",
            issue_body="Malformed URL may bypass validation.",
            use_rag=True,
        )
    )

    assert "rag_search" in tools.called
    assert [call.tool_name for call in response.tool_calls] == ["rag_query"]
    assert response.trace["called_tools"] == ["rag_query"]
    assert response.trace["deterministic_router"] is False
    assert response.trace["llm_provider"] == "fake-llm"
    assert "<rag_query>" not in response.answer
    assert "</function>" not in response.answer


def test_raw_malformed_function_closed_markup_is_not_returned_as_final_answer():
    tools = FakeChatTools()
    malformed_markup = '<rag_query>{"question":"URL parsing security","top_k":5}</function>'
    llm = FakeLlmProvider(
        first_content=malformed_markup,
        final_content=malformed_markup,
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(
        ChatRequest(message="Find guidance for URL parsing security.", use_rag=True)
    )

    assert [call.tool_name for call in response.tool_calls] == ["rag_query"]
    assert "Retrieved evidence / guidance" in response.answer
    assert "<rag_query>" not in response.answer
    assert "</function>" not in response.answer


def test_raw_xml_tool_tags_are_not_returned_when_final_answer_repeats_them():
    tools = FakeChatTools()
    raw_tool_markup = '<classify_issue>{"text":"Bug report"}</classify_issue>'
    llm = FakeLlmProvider(
        first_content=raw_tool_markup,
        final_content=raw_tool_markup,
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(ChatRequest(message="Please triage this bug.", use_rag=False))

    assert "<classify_issue>" not in response.answer
    assert "Likely label: bug" in response.answer
    assert [call.tool_name for call in response.tool_calls] == ["classify_issue"]


def test_invalid_xml_tool_json_is_handled_safely():
    tools = FakeChatTools()
    llm = FakeLlmProvider(
        first_content='<classify_issue>{"text": "unterminated"</classify_issue>',
        final_content="Some tool evidence is unavailable.",
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(ChatRequest(message="Please triage this issue.", use_rag=False))

    assert "classify_issue" not in tools.called
    assert response.tool_calls[0].tool_name == "classify_issue"
    assert response.tool_calls[0].status == "error"
    assert "Invalid tool JSON" in response.tool_calls[0].error
    assert "<classify_issue>" not in response.answer


def test_unknown_xml_tool_name_is_rejected():
    tools = FakeChatTools()
    llm = FakeLlmProvider(
        first_content='<delete_repository>{"repo":"nodejs/node"}</delete_repository>',
        final_content="Some requested tool was unavailable.",
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(ChatRequest(message="Please triage this issue.", use_rag=False))

    assert tools.called == []
    assert response.tool_calls[0].tool_name == "delete_repository"
    assert response.tool_calls[0].status == "error"
    assert "Unknown chat tool" in response.tool_calls[0].error


def test_unknown_function_closed_xml_tool_name_is_rejected():
    tools = FakeChatTools()
    llm = FakeLlmProvider(
        first_content='<delete_repository>{"repo":"nodejs/node"}</function>',
        final_content="Some requested tool was unavailable.",
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(ChatRequest(message="Please triage this issue.", use_rag=False))

    assert tools.called == []
    assert response.tool_calls[0].tool_name == "delete_repository"
    assert response.tool_calls[0].status == "error"
    assert "Unknown chat tool" in response.tool_calls[0].error


def test_write_memory_is_explicit_not_automatic():
    tools = FakeChatTools()
    llm = FakeLlmProvider(
        [
            LlmToolCall(
                id="call-1",
                name="classify_issue",
                arguments={"text": "Feature request"},
            )
        ]
    )
    service = ChatService(tools=tools, llm_provider=llm)

    service.chat(
        ChatRequest(message="Please classify this issue.", use_rag=False),
        current_user=CurrentUser(email="user@example.com", role="user"),
    )

    assert "write_memory" not in tools.called
    assert tools.memory_writes == []


def test_write_memory_runs_when_llm_explicitly_calls_tool():
    tools = FakeChatTools()
    llm = FakeLlmProvider(
        [
            LlmToolCall(
                id="call-1",
                name="write_memory",
                arguments={"text": "User prefers concise triage answers.", "memory_type": "semantic"},
            )
        ]
    )
    service = ChatService(tools=tools, llm_provider=llm)

    response = service.chat(
        ChatRequest(message="Remember that I prefer concise triage answers.", use_rag=False),
        current_user=CurrentUser(email="user@example.com", role="user"),
    )

    assert [call.tool_name for call in response.tool_calls] == ["write_memory"]
    assert tools.memory_writes == ["User prefers concise triage answers."]

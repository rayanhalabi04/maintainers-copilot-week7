from app.domain.rag import RetrievedChunk


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
        return f"{prefix}, the most relevant evidence says: {text}"

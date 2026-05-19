import re

from app.domain.summarization import SummarizeRequest, SummarizeResponse


class SummarizationService:
    def summarize(self, request: SummarizeRequest) -> SummarizeResponse:
        text = request.text.strip()

        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        if not sentences:
            return SummarizeResponse(summary="")

        selected_sentences = sentences[: request.max_sentences]
        summary = " ".join(selected_sentences)

        return SummarizeResponse(summary=summary)
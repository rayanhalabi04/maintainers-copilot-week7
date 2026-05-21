import re
import uuid
from collections.abc import Iterable

from app.domain.rag import RagChunk, RagDocument


def stable_id(*parts: object) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(str(part) for part in parts)))


class ParentChildChunker:
    def chunk_document(self, document: RagDocument) -> list[RagChunk]:
        if document.source_type == "issue":
            pieces = self._chunk_issue(document)
        else:
            pieces = self._chunk_markdown(document.text)
        chunks = []
        for _piece_index, (text, metadata) in enumerate(pieces):
            clean_text = text.strip()
            if not clean_text or self._is_tiny_useless_chunk(clean_text):
                continue
            index = len(chunks)
            chunks.append(
                RagChunk(
                    id=stable_id(document.id, index, clean_text[:80]),
                    document_id=document.id,
                    source_type=document.source_type,
                    source_id=document.source_id,
                    title=document.title,
                    text=clean_text,
                    chunk_index=index,
                    url=document.url,
                    metadata={**document.metadata, **metadata, "chunk_index": index},
                )
            )
        return chunks

    def _chunk_issue(self, document: RagDocument) -> list[tuple[str, dict[str, object]]]:
        parts = []
        sections = re.split(r"\n(?=##\s+)", document.text)
        if len(sections) <= 1:
            sections = re.split(r"\n(?=#+\s+|\*\*[^*]+\*\*:)", document.text)

        pending_issue_body = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue
            lowered = section.lower()
            if "issue body" in lowered and len(section.split()) <= 220:
                pending_issue_body = section
                continue
            if "maintainer" in lowered and pending_issue_body:
                combined = f"{pending_issue_body}\n\n{section}"
                parts.extend(self._split_long_text(combined, {"section": "issue and maintainer answer"}))
                pending_issue_body = ""
                continue
            name = "final maintainer answer" if "maintainer" in lowered else "body"
            parts.extend(self._split_long_text(section, {"section": name}))
        if pending_issue_body:
            parts.extend(self._split_long_text(pending_issue_body, {"section": "body"}))
        return parts

    def _chunk_markdown(self, text: str) -> list[tuple[str, dict[str, object]]]:
        blocks = re.split(r"(?m)(?=^#{1,3}\s+)", text)
        chunks = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", block)
            heading = heading_match.group(2).strip() if heading_match else None
            chunks.extend(self._split_long_text(block, {"section_heading": heading}))
        return chunks

    def _split_long_text(
        self, text: str, metadata: dict[str, object], max_words: int = 160
    ) -> Iterable[tuple[str, dict[str, object]]]:
        code_blocks = re.split(r"(```.*?```)", text, flags=re.DOTALL)
        for block in code_blocks:
            if not block.strip():
                continue
            if block.startswith("```"):
                yield block, {**metadata, "content_type": "code"}
                continue
            words = block.split()
            for start in range(0, len(words), max_words):
                yield " ".join(words[start : start + max_words]), metadata

    def _is_tiny_useless_chunk(self, text: str) -> bool:
        words = text.split()
        if len(words) >= 8:
            return False
        if text.startswith("#") and len(words) >= 3:
            return False
        return True

import re

from app.domain.ner import Entity, NerRequest, NerResponse


class NerService:
    def extract_entities(self, request: NerRequest) -> NerResponse:
        text = request.text
        entities: list[Entity] = []

        patterns = {
            "FILE_PATH": r"([A-Za-z0-9_\-/\.]+\.(js|ts|json|md|py|yml|yaml|txt))",
            "ERROR_CODE": r"\b[A-Z_]{3,}\b",
            "VERSION": r"\bv?\d+\.\d+(\.\d+)?\b",
            "URL": r"https?://[^\s]+",
            "GITHUB_REF": r"#\d+",
            "FUNCTION": r"\b[a-zA-Z_][a-zA-Z0-9_]+\(\)",
        }

        for label, pattern in patterns.items():
            for match in re.finditer(pattern, text):
                entities.append(
                    Entity(
                        text=match.group(0),
                        label=label,
                        start=match.start(),
                        end=match.end(),
                    )
                )

        return NerResponse(entities=entities)
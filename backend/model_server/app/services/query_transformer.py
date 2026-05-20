import re


CODE_ENTITY_PATTERN = re.compile(
    r"(\b[\w.-]+/[\w./-]+\b|\b[A-Z][A-Za-z0-9_]*(?:Error|Exception)\b|"
    r"\b[a-zA-Z_][\w]*\(\)|\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*\b|"
    r"\bv?\d+\.\d+(?:\.\d+)?\b|\b[a-z][a-z0-9_-]+(?:\.[a-z0-9_-]+)+\b)"
)


class QueryTransformer:
    def rewrite(self, query: str) -> tuple[str, list[str]]:
        cleaned = " ".join(query.strip().split())
        entities = list(dict.fromkeys(match.group(0) for match in CODE_ENTITY_PATTERN.finditer(cleaned)))
        if entities:
            return f"{cleaned} {' '.join(entities)}", entities
        return cleaned, []

    def generate_query_variants(self, query: str) -> tuple[list[str], list[str]]:
        cleaned, entities = self.rewrite(query)
        variants = [
            " ".join(query.strip().split()),
            f"{cleaned} GitHub issue Node.js",
            f"{cleaned} resolved issue maintainer answer",
        ]
        if entities:
            variants.append(f"{cleaned} technical tokens {' '.join(entities)}")
        return list(dict.fromkeys(variant for variant in variants if variant)), entities

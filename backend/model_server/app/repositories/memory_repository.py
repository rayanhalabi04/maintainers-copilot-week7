import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.memory import AuditLogRecord, MemoryRecord, MemoryType
from app.infra.redaction import redact_obj, redact_text


class LocalDemoMemoryRepository:
    """Local/demo JSON memory store; replace with Redis + Postgres/pgvector later."""

    def __init__(self, store_dir: Path | None = None) -> None:
        configured_dir = os.getenv("MEMORY_STORE_DIR")
        self.store_dir = store_dir or Path(configured_dir or "data/memory")
        self.memories_path = self.store_dir / "demo_memories.json"
        self.audit_log_path = self.store_dir / "audit_log.jsonl"
        self._ensure_store()

    def write_memory(
        self,
        user_email: str,
        text: str,
        memory_type: MemoryType,
        metadata: dict,
    ) -> MemoryRecord:
        memories = self._load_memories()
        record = MemoryRecord(
            memory_id=str(uuid4()),
            user_email=user_email,
            memory_type=memory_type,
            text=redact_text(text),
            metadata=redact_obj(metadata),
            created_at=self._now(),
        )
        memories.append(record.model_dump(mode="json"))
        self._save_memories(memories)
        return record

    def list_memories(self, user_email: str, limit: int = 10) -> list[MemoryRecord]:
        records = [
            MemoryRecord(**memory)
            for memory in self._load_memories()
            if memory.get("user_email") == user_email
        ]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records[:limit]

    def search_memories(
        self,
        user_email: str,
        query: str | None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        if not query:
            return self.list_memories(user_email, limit=limit)

        query_terms = [term for term in query.lower().split() if term]
        scored: list[tuple[int, MemoryRecord]] = []
        for memory in self.list_memories(user_email, limit=1000):
            haystack = f"{memory.text} {json.dumps(memory.metadata, sort_keys=True)}".lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score:
                scored.append((score, memory))

        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [memory for _score, memory in scored[:limit]]

    def write_audit(
        self,
        actor: str,
        action: str,
        target_type: str,
        target_id: str,
        metadata: dict,
    ) -> AuditLogRecord:
        record = AuditLogRecord(
            audit_id=str(uuid4()),
            actor=actor,
            action=action,
            target_type=target_type,
            target_id=target_id,
            timestamp=self._now(),
            metadata=redact_obj(metadata),
        )
        with self.audit_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return record

    def list_audit(self, limit: int = 50) -> list[AuditLogRecord]:
        if not self.audit_log_path.exists():
            return []
        records = []
        with self.audit_log_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(AuditLogRecord(**json.loads(line)))
        records.sort(key=lambda record: record.timestamp, reverse=True)
        return records[:limit]

    def _ensure_store(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if not self.memories_path.exists():
            self._save_memories([])
        if not self.audit_log_path.exists():
            self.audit_log_path.touch()

    def _load_memories(self) -> list[dict]:
        with self.memories_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save_memories(self, memories: list[dict]) -> None:
        with self.memories_path.open("w", encoding="utf-8") as file:
            json.dump(memories, file, indent=2, sort_keys=True)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

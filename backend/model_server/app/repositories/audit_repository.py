import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.domain.audit import AuditEvent, AuditEventCreate


class AuditEventRepository:
    def __init__(self, path: Path | None = None, database_url: str | None = None) -> None:
        configured_path = os.getenv("AUDIT_EVENTS_PATH")
        self.database_url = database_url or os.getenv("AUDIT_DATABASE_URL")
        self.path = path or Path(configured_path or "data/audit/audit_events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def write_event(self, event: AuditEventCreate) -> AuditEvent:
        record = AuditEvent(
            audit_id=str(uuid4()),
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            **event.model_dump(mode="json"),
        )
        if self.database_url:
            self._write_event_to_database(record)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
        return record

    def list_events(self, limit: int = 50) -> list[AuditEvent]:
        records = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(AuditEvent(**json.loads(line)))
        records.sort(key=lambda record: record.timestamp, reverse=True)
        return records[:limit]

    def _write_event_to_database(self, record: AuditEvent) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into audit_events (id, actor, action, target_type, target_id, metadata, created_at)
                    values (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        record.audit_id,
                        record.actor,
                        record.action,
                        record.target_type,
                        record.target_id,
                        json.dumps(record.metadata),
                        record.timestamp,
                    ),
                )

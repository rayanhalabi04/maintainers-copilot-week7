# Runbook

## Configure Postgres

Set `DATABASE_URL` for Postgres with pgvector installed:

```bash
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/maintainers_copilot
```

Without `DATABASE_URL`, the app uses `data/rag_store.json` for local smoke
testing.

## Run Migrations

```bash
uv run alembic upgrade head
```

## Ingest The RAG Corpus

```bash
uv run python scripts/ingest_rag_corpus.py
```

The script reads `docs/` when present and falls back to
`data/rag/sample_docs/`. It reads resolved issues from
`data/rag/resolved_issues_sample.jsonl` unless `--issues-file` is provided.

## Start The Server

From the model server directory:

```bash
cd backend/model_server
uv run uvicorn app.main:app --reload --port 8001
```

Or from the repo root:

```bash
PYTHONPATH=backend/model_server uv run uvicorn app.main:app --reload --port 8001
```

## Test RAG

```bash
curl -X POST http://localhost:8001/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How was the login token error fixed before?","top_k":5}'
```

With a metadata filter:

```bash
curl -X POST http://localhost:8001/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How was the JWTDecodeError login issue fixed before?","top_k":5,"filters":{"source_type":"issue"}}'
```

## Run Tests

```bash
uv run pytest
```

## Run RAG Evals

```bash
uv run python scripts/eval_rag.py
```

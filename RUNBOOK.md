# Runbook

## Environment

Set a GitHub token before fetching fresh issues:

```bash
export GITHUB_TOKEN=...
```

The token is used only for GitHub API requests and must not be committed. If no
token is set, `scripts/fetch_node_resolved_issues.py` can build from the local
`data/raw/resolved_issues_raw.json` file when that file exists.

Useful RAG environment variables:

```bash
export RAG_FORCE_LOCAL_STORE=false
export RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export RAG_RERANKER_ENABLED=true
```

Database URLs differ by where the command runs:

```bash
# Inside Docker Compose containers
export DATABASE_URL=postgresql://maintainers:maintainers@db:5432/maintainers_copilot

# From the Mac terminal when Postgres is remapped to localhost:5433
export DATABASE_URL=postgresql://maintainers:maintainers@localhost:5433/maintainers_copilot
```

For fast local/dev fallback only:

```bash
export RAG_FORCE_LOCAL_STORE=true
export RAG_LOCAL_STORE_PATH=data/rag_store.json
```

## Build The Real Node.js RAG Corpus

Fetch or copy selected Node.js docs:

```bash
uv run python3 scripts/fetch_node_docs.py
```

If `external/node` exists, the script copies docs from there. Otherwise it
downloads selected files from GitHub raw URLs.

Fetch closed Node.js issues with maintainer comments:

```bash
uv run python3 scripts/fetch_node_resolved_issues.py --target-count 250
```

This writes:

`data/rag/final_issues/node_resolved_issues.jsonl`

Build normalized documents and debug chunks:

```bash
uv run python3 scripts/build_final_rag_corpus.py
```

This writes:

- `data/rag/processed/rag_documents.jsonl`
- `data/rag/processed/rag_chunks.jsonl`

## Start Infrastructure

```bash
docker compose up -d db redis minio vault
docker compose run --rm migrate
```

## Ingest Into pgvector

```bash
export DATABASE_URL=postgresql://maintainers:maintainers@localhost:5433/maintainers_copilot
uv run python3 scripts/ingest_final_rag_corpus.py
```

The script uses `DATABASE_URL` and pgvector by default. It uses local JSON only
when `RAG_FORCE_LOCAL_STORE=true`.

## Verify Real Sources

Start the backend:

```bash
PYTHONPATH=backend/model_server uv run uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Or with Docker:

```bash
docker compose up --build model_server
```

Query RAG:

```bash
curl -s http://localhost:8001/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Where are Node.js filesystem APIs documented?","top_k":5,"filters":{"repo":"nodejs/node"}}'
```

The response sources should point to real Node.js URLs such as:

- `https://github.com/nodejs/node/blob/main/doc/api/fs.md`
- `https://github.com/nodejs/node/issues/...`

They should not point to:

- `data/rag/sample_docs/login.md`
- `github.com/example/maintainers-copilot`

## Run Tests And Evals

```bash
uv run pytest
export DATABASE_URL=postgresql://maintainers:maintainers@localhost:5433/maintainers_copilot
export RAG_FORCE_LOCAL_STORE=false
uv run python3 scripts/eval_rag.py
```

`scripts/eval_rag.py` writes `eval_report.json` and exits non-zero if advanced
RAG falls below any threshold in `eval_thresholds.yaml`.

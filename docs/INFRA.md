# Docker Infrastructure

Minimal local Docker infrastructure for the Week 7 Maintainer's Copilot.

## Start

```bash
cp .env.example .env
docker compose config
docker compose up --build
```

Run migrations explicitly again when you want to re-apply/check the Postgres/pgvector RAG schema:

```bash
docker compose run --rm migrate
```

## URLs

- Backend API: http://localhost:8001
- Streamlit internal app: http://localhost:8501
- React widget dev server: http://localhost:5173
- Host demo: http://localhost:8080
- MinIO console: http://localhost:9001
- Vault dev server: http://localhost:8200
- Postgres: localhost:5432
- Redis: localhost:6379

## Artifact Upload

With MinIO running:

```bash
python3 scripts/upload_artifacts_to_minio.py
```

The script uploads selected model artifacts and `eval_report.json` to the
`maintainers-copilot-artifacts` bucket.

## What Is Real

- Compose starts backend API, Streamlit, widget, host demo, Postgres/pgvector, Redis, MinIO, Vault, and a migration job profile.
- `GET /widget.js` points the iframe at the local widget dev server.
- MinIO has a manual artifact upload script.
- Minimal Redis, MinIO, and Vault adapter modules exist and fail safely outside Docker.

## What Is Still Demo / Not Fully Integrated

- Memory remains local JSON-backed by default. Redis is present for future short-term memory.
- RAG uses Postgres/pgvector when `DATABASE_URL` is set and `RAG_FORCE_LOCAL_STORE=false`.
  The local JSON store remains as a dev fallback only.
- MinIO is available, but artifact upload is manual and not wired into CI.
- Vault runs in dev mode, but the app does not require Vault secrets at boot.
- The Docker stack is for local development, not a production deployment.

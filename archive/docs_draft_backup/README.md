# Maintainer's Copilot

## Issue Classification

This repo compares three classifiers on the same processed GitHub issue splits:

- Classical ML baseline: TF-IDF plus logistic regression
- Fine-tuned transformer: `distilbert-base-uncased`
- LLM baseline: OpenAI-compatible chat completion API, when an API key is available

The four labels are `bug`, `feature`, `docs`, and `question`.

### Install

This project is pinned for macOS Intel with Python 3.12:

```bash
uv sync
```

The Python range is intentionally `>=3.12,<3.13` because the transformer stack uses `torch==2.2.2` and `numpy<2`. Leaving the project open to future Python versions can make uv solve incompatible NumPy and pandas combinations.

### Prepare The Dataset

```bash
uv run python backend/scripts/fetch_github_issues.py
uv run python backend/scripts/prepare_dataset.py
```

The processed splits are written to:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

Each row contains `id`, `number`, `text`, `label`, `created_at`, and `url`.

### Train The Classical Baseline

```bash
uv run python backend/scripts/train_classical.py
```

Outputs:

- `artifacts/classical_classifier.joblib`
- `reports/classical_metrics.json`

### Train The Transformer

```bash
uv run python backend/scripts/train_transformer.py
```

Outputs:

- `artifacts/transformer_classifier/`
- `artifacts/transformer_classifier/model_card.md`
- `reports/transformer_metrics.json`
- `reports/transformer_confusion_matrix.json`

For a faster smoke test, use limits:

```bash
uv run python backend/scripts/train_transformer.py --max-train-samples 64 --max-eval-samples 64 --epochs 1
```

For a fuller presentation run, increase epochs:

```bash
uv run python backend/scripts/train_transformer.py --epochs 3
```

### Evaluate The LLM Baseline

Without an API key, this command exits cleanly and writes a skipped report:

```bash
uv run python backend/scripts/evaluate_llm_baseline.py
```

To run it with an OpenAI-compatible API:

```bash
OPENAI_API_KEY=... uv run python backend/scripts/evaluate_llm_baseline.py
```

Optional settings:

- `LLM_MODEL`, default `gpt-4o-mini`
- `LLM_API_BASE`, default `https://api.openai.com/v1`
- `LLM_BASELINE_LIMIT`, useful for a small cost-controlled sample
- `LLM_INPUT_COST_PER_1M` and `LLM_OUTPUT_COST_PER_1M`, used for estimated cost

Output:

- `reports/llm_baseline_metrics.json`

### Generate The Combined Comparison

```bash
uv run python backend/scripts/generate_classification_comparison.py
```

Output:

- `reports/classification_comparison.json`

The comparison report includes accuracy, macro-F1, per-class F1, confusion matrices, latency, cost where available, and notes for the classical, transformer, and LLM baselines.

## Advanced RAG

The model server now exposes:

- `POST /rag/query`

The RAG pipeline uses parent-child chunking, dense retrieval, sparse retrieval,
hybrid score merging, optional cross-encoder reranking, and grounded answer
generation from retrieved sources.

Default local settings:

- `RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`
- `RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `RAG_RERANKER_ENABLED=true`
- `RAG_TOP_K=5`
- `RAG_DENSE_TOP_K=20`
- `RAG_SPARSE_TOP_K=20`
- `RAG_HYBRID_ALPHA=0.6`

Run migrations when Postgres/pgvector is configured:

```bash
uv run alembic upgrade head
```

Ingest the sample or local corpus:

```bash
uv run python scripts/ingest_rag_corpus.py
```

Start the server from `backend/model_server`:

```bash
uv run uvicorn app.main:app --reload --port 8001
```

Query RAG:

```bash
curl -X POST http://localhost:8001/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How was the login token error fixed before?","top_k":5}'
```

See `DECISIONS.md`, `RUNBOOK.md`, and `EVALS.md` for architecture notes,
operations, and evaluation scaffolding.

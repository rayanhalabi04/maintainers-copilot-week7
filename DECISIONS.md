# Decisions

## Final RAG Corpus

The production RAG corpus targets `nodejs/node` and is built from two source
families:

- selected real Node.js markdown documentation
- closed Node.js GitHub issues that include maintainer comments

The final corpus lives under:

- `data/rag/final_docs/`
- `data/rag/final_issues/node_resolved_issues.jsonl`
- `data/rag/processed/rag_documents.jsonl`
- `data/rag/processed/rag_chunks.jsonl`

The old `data/rag_store.json` sample store remains available only as a local
development fallback when `RAG_FORCE_LOCAL_STORE=true`.

This corpus is intentionally mixed:

- Node.js docs answer stable API, contribution, security, build, and governance
  questions.
- Held-out closed issues with maintainer comments answer triage-style questions
  using prior maintainer behavior.

## Issue Leakage Control

RAG issue ingestion excludes issue numbers found in
`backend/model_server/artifacts/train.csv`. This keeps the final RAG issue
slice separate from classifier training data.

Closed issues without maintainer-quality comments are skipped. Maintainer
comments are selected with the GitHub `author_association` heuristic:

- `OWNER`
- `MEMBER`
- `COLLABORATOR`

## Storage

Production RAG retrieval uses Postgres with pgvector when `DATABASE_URL` is set
and `RAG_FORCE_LOCAL_STORE` is not true. The local JSON repository remains for
tests and demos.

pgvector was chosen because it keeps dense vectors, source metadata, and sparse
SQL/full-text search close to the API backend. That makes filtering by `repo`,
`path`, `labels`, and source type straightforward while still supporting vector
similarity search.

## Embeddings And Retrieval

The default embedding model is:

`sentence-transformers/all-MiniLM-L6-v2`

Retrieval uses:

- smart markdown/issue chunking
- query transformation with code/entity expansion
- dense vector search
- sparse keyword search
- hybrid score merging with `RAG_HYBRID_ALPHA`
- optional cross-encoder reranking
- metadata filtering by source type, labels, path, repo, and dates

If sentence-transformers or the reranker cannot load, the backend falls back
safely instead of crashing local tests.

## RAG Evaluation Decision

The 25-question RAG golden set currently reports:

- naive retrieval: `hit@5 = 0.68`, `MRR@10 = 0.581`
- advanced RAG: `hit@5 = 0.68`, `MRR@10 = 0.603`

The honest read is that advanced RAG does not improve hit@5 yet: the correct
source appears in the top 5 at the same rate. It does improve ranking quality:
the first correct result appears earlier on average, as shown by MRR@10 rising
from 0.581 to 0.603.

Generation quality is gated with a deterministic judge for CI:

- faithfulness checks answer support in retrieved contexts
- answer relevancy checks whether the answer addresses the question and expected
  source

Five examples are hand-labeled in `evals/rag_hand_labels.jsonl` and compared
against the deterministic judge to report agreement.

# Decisions

## Parent-Child Chunking

The RAG corpus uses parent-child chunking so retrieval can match precise child
chunks while answer generation can still inspect nearby parent or sibling
context. Documentation parents are markdown pages or major sections. Resolved
issue parents are full issues, with children for title, body, comments, and the
final maintainer answer.

## Postgres And pgvector

The durable store is Postgres with pgvector because the project already needs a
simple server-side retrieval path and pgvector keeps dense vectors, metadata,
and full-text search in one database. A JSON local fallback exists only for
local smoke tests when Postgres is not configured.

## Hybrid Retrieval

Retrieval combines dense vector similarity with sparse Postgres full-text
search. Dense search handles semantic matches; sparse search protects exact
tokens such as error names, paths, function names, and package versions.

## Hybrid Alpha

The default `RAG_HYBRID_ALPHA` is `0.6`, giving slightly more weight to dense
semantic retrieval while keeping exact lexical matches influential.

## Optional Reranking

Cross-encoder reranking is configurable because it can improve ordering but
adds model load time and compute. Retrieval still works when
`RAG_RERANKER_ENABLED=false`.

## Embeddings

The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`, chosen
because it is small enough for local development and produces 384-dimensional
vectors suitable for pgvector.

## Evaluation TODO

The placeholder golden set must be replaced with real maintainer questions and
verified chunk IDs. After that, `hit@5`, `MRR@10`, answer groundedness, and
source precision should be measured before tuning alpha, top-k, or reranking.

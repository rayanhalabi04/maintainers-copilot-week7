# Evals

RAG eval scaffolding lives in:

- `data/evals/rag_golden.jsonl`
- `scripts/eval_rag.py`
- `eval_thresholds.yaml`

The current golden file contains placeholder examples. Replace
`ground_truth_chunk_ids` with real chunk IDs after ingesting production docs and
resolved issues.

Run:

```bash
uv run python scripts/eval_rag.py
```

The script writes `eval_report.json` with:

- `hit_at_5`
- `mrr_at_10`

Current thresholds are intentionally non-zero placeholders so the pipeline has
a visible quality gate before the real golden set is built.

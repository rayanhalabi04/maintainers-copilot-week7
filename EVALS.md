# Evaluations

## RAG Golden Set

The active RAG golden set is:

`evals/rag_golden_set.jsonl`

It contains 25 Node.js documentation questions with ideal answers and stable
ground-truth source paths such as `README.md` and `doc/api/fs.md`.

## Metrics

Run:

```bash
uv run python3 scripts/eval_rag.py
```

The required retrieval metrics are:

- `hit@5`
- `MRR@10`

`hit@5` asks: did one of the correct source chunks appear in the top 5?

`MRR@10` asks: how high was the first correct source ranked in the top 10?
Higher MRR means the right evidence is closer to the top of the result list.

The required generation metrics are:

- `faithfulness`: whether answer content is supported by retrieved contexts
- `answer_relevancy`: whether the answer addresses the question and expected source

The CI-safe judge is a deterministic token-overlap judge. It does not require
API keys. A future frozen LLM judge or RAGAS judge can replace it, but the
current fallback keeps CI reproducible.

Current production retrieval numbers on the 25-question set:

| Mode | hit@5 | MRR@10 |
| --- | ---: | ---: |
| Naive dense retrieval | 0.68 | 0.581 |
| Advanced RAG | 0.68 | 0.603 |

The advanced pipeline keeps hit@5 flat at 0.68, but improves ranking quality:
MRR@10 increases from 0.581 to 0.603.

## Hand-Labeled Judge Agreement

The hand-label file is:

`evals/rag_hand_labels.jsonl`

It contains 5 examples from the 25-question golden set. Each record has a human
faithfulness label, a human answer relevancy label, the expected source, and a
short note. `scripts/eval_rag.py` scores the same 5 examples with the
deterministic judge, thresholds the continuous scores into pass/fail labels,
and reports:

- `faithfulness_agreement`
- `answer_relevancy_agreement`
- `overall_judge_agreement`

## RAG Gates

RAG thresholds live in `eval_thresholds.yaml`:

- `hit_at_5 >= 0.60`
- `mrr_at_10 >= 0.50`
- `faithfulness >= 0.55`
- `answer_relevancy >= 0.20`

`scripts/eval_rag.py` exits non-zero if any required advanced RAG metric falls
below threshold, so CI fails on retrieval or generation regressions.

## Classification Evals

The classifier eval remains separate:

```bash
uv run python3 scripts/eval_classification.py
```

RAG issue ingestion excludes classifier train issue numbers from
`backend/model_server/artifacts/train.csv`.

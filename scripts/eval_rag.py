import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "model_server"))

from app.services.rag_service import RagService  # noqa: E402
from app.infra.eval_thresholds import ThresholdValidationError, load_rag_thresholds  # noqa: E402


EVAL_REPORT_PATH = ROOT / "eval_report.json"
THRESHOLDS_PATH = ROOT / "eval_thresholds.yaml"
HAND_LABELS_PATH = ROOT / "evals" / "rag_hand_labels.jsonl"
CORPUS_MANIFEST_PATH = ROOT / "data" / "rag" / "processed" / "manifest.json"
DEFAULT_GOLDEN_PATHS = [
    ROOT / "evals" / "rag_golden_set.jsonl",
    ROOT / "data" / "evals" / "rag_golden.jsonl",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "where",
    "which",
    "who",
    "with",
    "should",
    "does",
    "do",
    "i",
}


def _load_examples(golden_path: Path) -> list[dict]:
    examples = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    placeholder_rows = []
    for index, example in enumerate(examples, start=1):
        chunk_ids = example.get("ground_truth_chunk_ids") or example.get("ground_truth_chunks") or []
        if any(str(chunk_id).startswith("placeholder") for chunk_id in chunk_ids):
            placeholder_rows.append(index)
    if placeholder_rows:
        rows = ", ".join(str(row) for row in placeholder_rows)
        raise SystemExit(
            "RAG golden contains placeholder chunk IDs. "
            f"Fix data/evals/rag_golden.jsonl before evaluation. Rows: {rows}"
        )
    return examples


def _score(
    examples: list[dict],
    retrieve: Callable[[str], list],
) -> dict[str, float | int]:
    hit5 = 0
    reciprocal_sum = 0.0

    for example in examples:
        chunks = retrieve(example["question"])
        retrieved_keys = [_retrieval_keys(chunk) for chunk in chunks]
        truth = set(
            example.get("ground_truth_chunk_ids")
            or example.get("ground_truth_chunks")
            or example.get("ground_truth_sources")
            or []
        )
        if any(truth & keys for keys in retrieved_keys[:5]):
            hit5 += 1
        for rank, keys in enumerate(retrieved_keys[:10], start=1):
            if truth & keys:
                reciprocal_sum += 1 / rank
                break

    return {
        "examples": len(examples),
        "hit_at_5": hit5 / len(examples) if examples else 0.0,
        "mrr_at_10": reciprocal_sum / len(examples) if examples else 0.0,
    }


def _evaluate_advanced(examples: list[dict], service: RagService) -> dict[str, float | int]:
    hit5 = 0
    reciprocal_sum = 0.0
    faithfulness_scores = []
    relevancy_scores = []
    per_example = []

    for example in examples:
        chunks, _debug = service.retrieve_advanced(example["question"], top_k=10)
        answer = service._answer(example["question"], chunks)
        retrieved_keys = [_retrieval_keys(chunk) for chunk in chunks]
        truth = set(
            example.get("ground_truth_chunk_ids")
            or example.get("ground_truth_chunks")
            or example.get("ground_truth_sources")
            or []
        )
        hit_at_5 = any(truth & keys for keys in retrieved_keys[:5])
        reciprocal_rank = 0.0
        if hit_at_5:
            hit5 += 1
        for rank, keys in enumerate(retrieved_keys[:10], start=1):
            if truth & keys:
                reciprocal_rank = 1 / rank
                reciprocal_sum += reciprocal_rank
                break
        faithfulness_scores.append(_score_faithfulness(answer, chunks))
        relevancy_scores.append(_score_answer_relevancy(answer, example, chunks))
        per_example.append(
            {
                "question": example["question"],
                "user_input": example["question"],
                "reference": example.get("ideal_answer"),
                "ideal_answer": example.get("ideal_answer"),
                "generated_response": answer,
                "ground_truth_chunks": list(truth),
                "retrieved_contexts_top5": [
                    _context_detail(chunk, rank)
                    for rank, chunk in enumerate(chunks[:5], start=1)
                ],
                "hit_at_5": hit_at_5,
                "reciprocal_rank": reciprocal_rank,
            }
        )

    return {
        "examples": len(examples),
        "hit_at_5": hit5 / len(examples) if examples else 0.0,
        "mrr_at_10": reciprocal_sum / len(examples) if examples else 0.0,
        "faithfulness": _mean(faithfulness_scores),
        "answer_relevancy": _mean(relevancy_scores),
        "judge": "deterministic_token_overlap_v1",
        "per_example": per_example,
    }


def _load_thresholds(path: Path) -> dict[str, float]:
    try:
        return load_rag_thresholds(path)
    except ThresholdValidationError as exc:
        raise SystemExit(str(exc)) from exc


def _retrieval_keys(chunk) -> set[str]:
    metadata = getattr(chunk, "metadata", {}) or {}
    parent_metadata = metadata.get("parent_metadata") or {}
    keys = {
        getattr(chunk, "chunk_id", ""),
        getattr(chunk, "document_id", ""),
        getattr(chunk, "source_id", ""),
        str(metadata.get("path") or ""),
        str(parent_metadata.get("path") or ""),
    }
    return {key for key in keys if key}


def _context_detail(chunk, rank: int) -> dict:
    text = str(getattr(chunk, "text", ""))
    return {
        "rank": rank,
        "source_type": getattr(chunk, "source_type", None),
        "title": getattr(chunk, "title", None),
        "url": getattr(chunk, "url", None),
        "chunk_id": getattr(chunk, "chunk_id", None),
        "score": float(getattr(chunk, "score", 0.0) or 0.0),
        "text_preview": text[:500],
    }


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_./-]*", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _score_faithfulness(answer: str, chunks: list) -> float:
    answer_tokens = _tokens(_strip_answer_boilerplate(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = _tokens(" ".join(getattr(chunk, "text", "") for chunk in chunks[:5]))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _score_answer_relevancy(answer: str, example: dict, chunks: list | None = None) -> float:
    source_text = ""
    if chunks:
        source_text = " ".join(
            " ".join(
                [
                    str(getattr(chunk, "title", "")),
                    str(getattr(chunk, "source_id", "")),
                    str((getattr(chunk, "metadata", {}) or {}).get("path") or ""),
                ]
            )
            for chunk in chunks[:3]
        )
    answer_tokens = _tokens(f"{answer} {source_text}")
    target_tokens = _tokens(
        " ".join(
            [
                str(example.get("question") or ""),
                str(example.get("ideal_answer") or ""),
                " ".join(example.get("ground_truth_chunks") or []),
                " ".join(example.get("ground_truth_sources") or []),
            ]
        )
    )
    if not answer_tokens or not target_tokens:
        return 0.0
    precision = len(answer_tokens & target_tokens) / len(answer_tokens)
    recall = len(answer_tokens & target_tokens) / len(target_tokens)
    if precision + recall == 0:
        return 0.0
    score = 2 * precision * recall / (precision + recall)
    if chunks:
        truth = set(
            example.get("ground_truth_chunk_ids")
            or example.get("ground_truth_chunks")
            or example.get("ground_truth_sources")
            or []
        )
        retrieved_keys = [_retrieval_keys(chunk) for chunk in chunks[:5]]
        if any(truth & keys for keys in retrieved_keys):
            score = max(score, 0.75)
    return score


def _strip_answer_boilerplate(answer: str) -> str:
    markers = [
        "the most relevant evidence says:",
        "based on the retrieved project evidence,",
        "based on previous resolved issues,",
    ]
    lowered = answer.lower()
    for marker in markers:
        if marker in lowered:
            return answer[lowered.index(marker) + len(marker) :]
    return answer


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _load_hand_labels(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _evaluate_judge_agreement(
    labels: list[dict],
    service: RagService,
    thresholds: dict[str, float],
) -> dict[str, float | int | str]:
    if not labels:
        return {
            "examples": 0,
            "faithfulness_agreement": 0.0,
            "answer_relevancy_agreement": 0.0,
            "overall_judge_agreement": 0.0,
            "judge": "deterministic_token_overlap_v1",
        }

    faithfulness_matches = 0
    relevancy_matches = 0
    total = len(labels)
    for label in labels:
        chunks, _debug = service.retrieve_advanced(label["question"], top_k=10)
        answer = service._answer(label["question"], chunks)
        faithfulness_label = int(
            _score_faithfulness(answer, chunks) >= thresholds["faithfulness"]
        )
        relevancy_label = int(
            _score_answer_relevancy(answer, label, chunks) >= thresholds["answer_relevancy"]
        )
        if faithfulness_label == int(label["human_faithfulness_label"]):
            faithfulness_matches += 1
        if relevancy_label == int(label["human_answer_relevancy_label"]):
            relevancy_matches += 1

    faithfulness_agreement = faithfulness_matches / total
    relevancy_agreement = relevancy_matches / total
    return {
        "examples": total,
        "faithfulness_agreement": faithfulness_agreement,
        "answer_relevancy_agreement": relevancy_agreement,
        "overall_judge_agreement": (faithfulness_matches + relevancy_matches) / (2 * total),
        "judge": "deterministic_token_overlap_v1",
    }


def _load_corpus_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {
        "documents": data.get("documents"),
        "chunks": data.get("chunks"),
        "doc_documents": data.get("doc_documents"),
        "issue_documents": data.get("issue_documents"),
        "repo": data.get("repo"),
    }


def _default_golden_path() -> Path:
    for path in DEFAULT_GOLDEN_PATHS:
        if path.exists():
            return path
    return DEFAULT_GOLDEN_PATHS[0]


def _load_existing_report(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path.relative_to(ROOT)} is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _enforce_thresholds(metrics: dict, thresholds: dict[str, float]) -> None:
    advanced = metrics["advanced"]
    failures = []
    for key in ("hit_at_5", "mrr_at_10", "faithfulness", "answer_relevancy"):
        if advanced[key] < thresholds[key]:
            failures.append(f"advanced.{key}={advanced[key]:.4f} below threshold {thresholds[key]:.4f}")
    if failures:
        raise SystemExit("RAG eval failed: " + "; ".join(failures))


def main() -> None:
    golden_path = _default_golden_path()
    examples = _load_examples(golden_path)
    service = RagService()

    thresholds = _load_thresholds(THRESHOLDS_PATH)
    hand_labels = _load_hand_labels(HAND_LABELS_PATH)
    rag_report = {
        "naive": _score(
            examples,
            lambda question: service.retrieve_naive(question, top_k=10),
        ),
        "advanced": _evaluate_advanced(examples, service),
        "judge_agreement": _evaluate_judge_agreement(hand_labels, service, thresholds),
        "thresholds": thresholds,
        "golden_path": str(golden_path.relative_to(ROOT)),
        "hand_labels_path": str(HAND_LABELS_PATH.relative_to(ROOT)),
        "corpus_summary": _load_corpus_summary(CORPUS_MANIFEST_PATH),
        "status": "pass",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _enforce_thresholds(rag_report, thresholds)
    except SystemExit:
        rag_report["status"] = "fail"
        report = _load_existing_report(EVAL_REPORT_PATH)
        report["rag"] = rag_report
        EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"rag": rag_report}, indent=2))
        raise
    report = _load_existing_report(EVAL_REPORT_PATH)
    report["rag"] = rag_report
    EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rag": rag_report}, indent=2))


if __name__ == "__main__":
    main()

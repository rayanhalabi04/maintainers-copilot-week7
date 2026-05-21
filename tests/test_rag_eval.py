import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.rag import RetrievedChunk
from scripts import eval_rag


def make_chunk(chunk_id: str = "chunk-1", source_id: str = "README.md") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_type="doc",
        source_id=source_id,
        title="Node.js README",
        url="https://github.com/nodejs/node/blob/main/README.md",
        score=0.9,
        text="The Node.js README points users to https://nodejs.org for downloads.",
        metadata={"path": source_id, "repo": "nodejs/node"},
    )


class FakeRagService:
    def retrieve_advanced(self, question: str, top_k: int = 10):
        return [make_chunk()], None

    def _answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return (
            "Based on the retrieved project evidence from Node.js README, "
            "the most relevant evidence says: The README points users to "
            "https://nodejs.org for downloads."
        )


def test_rag_thresholds_are_loaded_from_yaml(tmp_path: Path):
    path = tmp_path / "eval_thresholds.yaml"
    path.write_text(
        "rag:\n"
        "  hit_at_5: 0.60\n"
        "  mrr_at_10: 0.50\n"
        "  faithfulness: 0.55\n"
        "  answer_relevancy: 0.20\n"
        "classification_accuracy_min: 0.5\n"
        "classification_macro_f1_min: 0.5\n"
    )

    thresholds = eval_rag._load_thresholds(path)

    assert thresholds == {
        "hit_at_5": 0.60,
        "mrr_at_10": 0.50,
        "faithfulness": 0.55,
        "answer_relevancy": 0.20,
    }


def test_generation_metrics_are_added_to_advanced_eval():
    examples = [
        {
            "question": "Where does the Node.js README direct users who want to download Node.js?",
            "ideal_answer": "The README points users to https://nodejs.org for downloads.",
            "ground_truth_chunks": ["README.md"],
        }
    ]

    metrics = eval_rag._evaluate_advanced(examples, FakeRagService())

    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["faithfulness"] >= 0.55
    assert metrics["answer_relevancy"] >= 0.20
    assert metrics["judge"] == "deterministic_token_overlap_v1"


def test_advanced_eval_includes_per_example_context_details():
    examples = [
        {
            "question": "Where does the Node.js README direct users who want to download Node.js?",
            "ideal_answer": "The README points users to https://nodejs.org for downloads.",
            "ground_truth_chunks": ["README.md"],
        }
    ]

    metrics = eval_rag._evaluate_advanced(examples, FakeRagService())
    detail = metrics["per_example"][0]

    assert detail["question"] == examples[0]["question"]
    assert detail["reference"] == examples[0]["ideal_answer"]
    assert detail["generated_response"]
    assert detail["ground_truth_chunks"] == ["README.md"]
    assert detail["hit_at_5"] is True
    assert detail["reciprocal_rank"] == 1.0
    assert detail["retrieved_contexts_top5"][0]["chunk_id"] == "chunk-1"
    assert detail["retrieved_contexts_top5"][0]["rank"] == 1


def test_hand_label_agreement_is_computed():
    labels = [
        {
            "question": "Where does the Node.js README direct users who want to download Node.js?",
            "ideal_answer": "The README points users to https://nodejs.org for downloads.",
            "ground_truth_chunks": ["README.md"],
            "human_faithfulness_label": 1,
            "human_answer_relevancy_label": 1,
        }
    ]
    thresholds = {
        "hit_at_5": 0.60,
        "mrr_at_10": 0.50,
        "faithfulness": 0.55,
        "answer_relevancy": 0.20,
    }

    agreement = eval_rag._evaluate_judge_agreement(labels, FakeRagService(), thresholds)

    assert agreement["faithfulness_agreement"] == 1.0
    assert agreement["answer_relevancy_agreement"] == 1.0
    assert agreement["overall_judge_agreement"] == 1.0


def test_failing_threshold_raises_clear_error():
    metrics = {
        "advanced": {
            "hit_at_5": 0.59,
            "mrr_at_10": 0.51,
            "faithfulness": 0.80,
            "answer_relevancy": 0.50,
        }
    }
    thresholds = {
        "hit_at_5": 0.60,
        "mrr_at_10": 0.50,
        "faithfulness": 0.55,
        "answer_relevancy": 0.20,
    }

    with pytest.raises(SystemExit) as exc:
        eval_rag._enforce_thresholds(metrics, thresholds)

    assert "advanced.hit_at_5=0.5900 below threshold 0.6000" in str(exc.value)


def test_ci_safe_judge_fallback_works_without_api_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chunk = make_chunk()
    example = {
        "question": "Where does the Node.js README direct users who want to download Node.js?",
        "ideal_answer": "The README points users to https://nodejs.org for downloads.",
        "ground_truth_chunks": ["README.md"],
    }
    answer = "The README points users to https://nodejs.org for downloads."

    assert eval_rag._score_faithfulness(answer, [chunk]) > 0
    assert eval_rag._score_answer_relevancy(answer, example, [chunk]) >= 0.20

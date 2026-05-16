from __future__ import annotations

from fastapi.testclient import TestClient

from app.evaluation.models import ToolCallRecord
from app.evaluation.scorers import score_reference_metrics
from app.evaluation.service import EvaluationService


def test_reference_metrics_return_scores() -> None:
    metrics = score_reference_metrics("I agree with you.", "I agree with you.")

    assert metrics["bleu_score"] == 1.0
    assert metrics["rouge1_f1"] == 1.0
    assert metrics["rouge_l_f1"] == 1.0
    assert metrics["grammar_correction_quality"] == 1.0


def test_evaluation_service_computes_tool_and_task_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EVALUATION_ENABLED", "true")
    monkeypatch.setenv("EVALUATION_DATA_DIR", str(tmp_path))
    service = EvaluationService()

    record = service.evaluate_interaction(
        request_id="req-1",
        session_id="session-1",
        input_text="Correct this sentence: I am agree with you.",
        output_text="I agree with you.",
        expected_output="I agree with you.",
        reference_context="Correct English sentence: I agree with you.",
        model_name="test-model",
        provider="ollama",
        task_type="grammar_correction",
        tags=["english", "grammar"],
        latency_ms=123.4,
        input_tokens=10,
        output_tokens=5,
        tool_calls=[ToolCallRecord(name="rag_retrieval", status="success", latency_ms=8.0)],
    )

    assert record.metrics.task_success_rate == 1.0
    assert record.metrics.tool_calls_count == 1
    assert record.metrics.tool_call_success_rate == 1.0
    assert record.metrics.tool_call_error_rate == 0.0

    path = service.persist_interaction(record)
    assert path is not None
    assert path.exists()


def test_feedback_endpoint_updates_saved_interaction(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("EVALUATION_ENABLED", "true")
    monkeypatch.setenv("EVALUATION_DATA_DIR", str(tmp_path))

    service = EvaluationService()
    record = service.evaluate_interaction(
        request_id="feedback-1",
        session_id="session-1",
        input_text="hello",
        output_text="hello there",
        model_name="test-model",
        provider="ollama",
        task_type="conversation",
        latency_ms=50.0,
    )
    service.persist_interaction(record)

    from app.api import app

    response = TestClient(app).post(
        "/api/feedback",
        json={"request_id": "feedback-1", "score": 4, "comment": "Helpful"},
    )

    assert response.status_code == 200
    assert response.json() == {"request_id": "feedback-1", "saved": True}

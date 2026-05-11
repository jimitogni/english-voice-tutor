from __future__ import annotations

import os

from fastapi.testclient import TestClient

from app.config import load_config
from app.observability.langfuse_client import LangfuseTracer
from app.observability.privacy import hash_identifier, redact_text


def test_langfuse_tracer_disabled_does_not_crash(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    config = load_config()

    tracer = LangfuseTracer(config)
    trace = tracer.create_trace(
        name="test",
        request_id="request-id",
        session_id="session-id",
        user_id_hash=None,
        input_text="hello",
        metadata={},
    )
    tracer.update_trace(trace, output_text="world")
    tracer.flush()

    assert trace is None


def test_redaction_and_hash_helpers() -> None:
    assert hash_identifier("jimi") == hash_identifier("jimi")
    assert hash_identifier("jimi") != hash_identifier("other")
    assert redact_text("email me at person@example.com", redact=False) == "email me at [email]"
    assert redact_text("secret=abc123", redact=False) == "secret=[redacted]"
    assert redact_text("hello", redact=True) == "[redacted]"


def test_metrics_endpoint_returns_prometheus_format(monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("METRICS_ENABLED", "true")
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")

    from app.api import app

    response = TestClient(app).get("/api/metrics")

    assert response.status_code == 200
    assert "fastapi_requests_total" in response.text or "prometheus_client is not installed" in response.text


def teardown_module() -> None:
    for name in ["LANGFUSE_ENABLED", "METRICS_ENABLED", "PROMETHEUS_ENABLED"]:
        os.environ.pop(name, None)

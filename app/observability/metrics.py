from __future__ import annotations

from typing import Iterable

from fastapi.responses import PlainTextResponse, Response

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Histogram,
        generate_latest,
    )
except ImportError:  # pragma: no cover - only used before dependencies are installed.
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

    class CollectorRegistry:  # type: ignore[no-redef]
        pass

    class _NoopMetric:
        def labels(self, *_args):
            return self

        def inc(self, *_args):
            return None

        def observe(self, *_args):
            return None

    class Counter(_NoopMetric):  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class Histogram(_NoopMetric):  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    def generate_latest(_registry) -> bytes:  # type: ignore[no-redef]
        return b"# prometheus_client is not installed\n"

REGISTRY = CollectorRegistry()

FASTAPI_REQUESTS_TOTAL = Counter(
    "fastapi_requests_total",
    "Total FastAPI requests.",
    ["method", "path", "status_code"],
    registry=REGISTRY,
)
FASTAPI_REQUEST_LATENCY_SECONDS = Histogram(
    "fastapi_request_latency_seconds",
    "FastAPI request latency in seconds.",
    ["method", "path"],
    registry=REGISTRY,
)
FASTAPI_ERRORS_TOTAL = Counter(
    "fastapi_errors_total",
    "Total FastAPI request errors.",
    ["method", "path", "status_code"],
    registry=REGISTRY,
)

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests.",
    ["model", "provider", "status"],
    registry=REGISTRY,
)
LLM_REQUEST_LATENCY_SECONDS = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency in seconds.",
    ["model", "provider"],
    registry=REGISTRY,
)
LLM_ERRORS_TOTAL = Counter(
    "llm_errors_total",
    "Total LLM errors.",
    ["model", "provider"],
    registry=REGISTRY,
)
LLM_TOKENS_INPUT_TOTAL = Counter(
    "llm_tokens_input_total",
    "Total LLM input tokens, exact when available and estimated otherwise.",
    ["model", "provider", "source"],
    registry=REGISTRY,
)
LLM_TOKENS_OUTPUT_TOTAL = Counter(
    "llm_tokens_output_total",
    "Total LLM output tokens, exact when available and estimated otherwise.",
    ["model", "provider", "source"],
    registry=REGISTRY,
)

STT_REQUESTS_TOTAL = Counter(
    "stt_requests_total",
    "Total speech-to-text requests.",
    ["backend", "status"],
    registry=REGISTRY,
)
STT_REQUEST_LATENCY_SECONDS = Histogram(
    "stt_request_latency_seconds",
    "Speech-to-text request latency in seconds.",
    ["backend"],
    registry=REGISTRY,
)

TTS_REQUESTS_TOTAL = Counter(
    "tts_requests_total",
    "Total text-to-speech requests.",
    ["engine", "status"],
    registry=REGISTRY,
)
TTS_REQUEST_LATENCY_SECONDS = Histogram(
    "tts_request_latency_seconds",
    "Text-to-speech request latency in seconds.",
    ["engine"],
    registry=REGISTRY,
)


def observe_fastapi_request(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_seconds: float,
) -> None:
    status = str(status_code)
    FASTAPI_REQUESTS_TOTAL.labels(method, path, status).inc()
    FASTAPI_REQUEST_LATENCY_SECONDS.labels(method, path).observe(latency_seconds)
    if status_code >= 500:
        FASTAPI_ERRORS_TOTAL.labels(method, path, status).inc()


def observe_llm_call(
    *,
    model: str,
    provider: str,
    status: str,
    latency_seconds: float,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    token_source: str = "estimated",
) -> None:
    LLM_REQUESTS_TOTAL.labels(model, provider, status).inc()
    LLM_REQUEST_LATENCY_SECONDS.labels(model, provider).observe(latency_seconds)
    if status != "success":
        LLM_ERRORS_TOTAL.labels(model, provider).inc()
    if input_tokens is not None:
        LLM_TOKENS_INPUT_TOTAL.labels(model, provider, token_source).inc(max(input_tokens, 0))
    if output_tokens is not None:
        LLM_TOKENS_OUTPUT_TOTAL.labels(model, provider, token_source).inc(max(output_tokens, 0))


def observe_stt_call(*, backend: str, status: str, latency_seconds: float) -> None:
    STT_REQUESTS_TOTAL.labels(backend, status).inc()
    STT_REQUEST_LATENCY_SECONDS.labels(backend).observe(latency_seconds)


def observe_tts_call(*, engine: str, status: str, latency_seconds: float) -> None:
    TTS_REQUESTS_TOTAL.labels(engine, status).inc()
    TTS_REQUEST_LATENCY_SECONDS.labels(engine).observe(latency_seconds)


def metrics_response(enabled: bool) -> Response:
    if not enabled:
        return PlainTextResponse("# metrics disabled\n", media_type="text/plain")
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


def estimate_tokens_from_text(text: str | None) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_tokens_from_messages(messages: Iterable[dict[str, str]]) -> int:
    return sum(estimate_tokens_from_text(message.get("content", "")) for message in messages)

from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

from app.config import AppConfig, load_config
from app.observability.privacy import maybe_prompt, maybe_response, sanitize_messages

logger = logging.getLogger(__name__)


def langfuse_credentials_configured(config: AppConfig) -> bool:
    placeholders = {"", "replace_me", "change_me"}
    public_key = config.langfuse_public_key.strip()
    secret_key = config.langfuse_secret_key.strip()
    return public_key.lower() not in placeholders and secret_key.lower() not in placeholders


class LangfuseTracer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.enabled = config.langfuse_enabled
        self.client: Any | None = None
        if not self.enabled:
            return
        if not langfuse_credentials_configured(config):
            logger.warning("Langfuse is enabled but credentials are missing.")
            self.enabled = False
            return

        try:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                host=config.langfuse_host,
            )
        except Exception as exc:  # pragma: no cover - depends on optional service.
            logger.warning("Langfuse client initialization failed: %s", exc)
            self.enabled = False
            self.client = None

    def create_trace(
        self,
        *,
        name: str,
        request_id: str,
        session_id: str | None,
        user_id_hash: str | None,
        input_text: str | None,
        metadata: dict[str, Any],
    ) -> Any | None:
        if not self.enabled or self.client is None:
            return None
        try:
            return self.client.trace(
                id=request_id,
                name=name,
                session_id=session_id,
                user_id=user_id_hash,
                input=maybe_prompt(input_text, self.config),
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive integration.
            logger.debug("Langfuse trace creation failed: %s", exc)
            return None

    def update_trace(
        self,
        trace: Any | None,
        *,
        output_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if trace is None:
            return
        try:
            trace.update(output=maybe_response(output_text, self.config), metadata=metadata or {})
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse trace update failed: %s", exc)

    def log_span(
        self,
        trace: Any | None,
        *,
        name: str,
        input_value: Any | None = None,
        output_value: Any | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: float | None = None,
    ) -> None:
        if trace is None:
            return
        span = None
        try:
            span = trace.span(name=name, input=input_value, metadata=metadata or {})
            if output_value is not None:
                span.end(output=output_value)
            else:
                span.end()
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse span logging failed: %s", exc)
        finally:
            if started_at is not None and metadata is not None:
                metadata.setdefault("latency_ms", round((perf_counter() - started_at) * 1000, 2))

    def log_generation(
        self,
        trace: Any | None,
        *,
        name: str,
        model: str,
        messages: list[dict[str, str]],
        response_text: str | None,
        metadata: dict[str, Any],
        usage: dict[str, int] | None,
    ) -> None:
        if trace is None:
            return
        try:
            generation = trace.generation(
                name=name,
                model=model,
                input=sanitize_messages(messages, self.config),
                output=maybe_response(response_text, self.config),
                metadata=metadata,
                usage=usage,
            )
            generation.end()
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse generation logging failed: %s", exc)

    def score(
        self,
        *,
        trace_id: str,
        name: str,
        value: float | int,
        comment: str | None = None,
    ) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.score(trace_id=trace_id, name=name, value=value, comment=comment)
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse score logging failed: %s", exc)

    def flush(self) -> None:
        if not self.enabled or self.client is None:
            return
        try:
            self.client.flush()
        except Exception as exc:  # pragma: no cover
            logger.debug("Langfuse flush failed: %s", exc)


@lru_cache
def get_langfuse_tracer() -> LangfuseTracer:
    return LangfuseTracer(load_config())

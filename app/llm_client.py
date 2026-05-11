from __future__ import annotations

import json
import logging
from time import perf_counter
from collections.abc import Iterator
from typing import Any

import requests

from app.config import AppConfig, load_config
from app.observability import current_trace
from app.observability.langfuse_client import get_langfuse_tracer
from app.observability.logging import log_event
from app.observability.metrics import (
    estimate_tokens_from_messages,
    estimate_tokens_from_text,
    observe_llm_call,
)
from app.observability.privacy import sanitize_messages

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Base exception for local Ollama failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama server is unavailable."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model is not available locally."""


class OllamaClient:
    def __init__(self, config: AppConfig | None = None, timeout_seconds: int = 120) -> None:
        self.config = config or load_config()
        self.base_url = self.config.ollama_base_url.rstrip("/")
        self.model = self.config.ollama_model
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                "Could not reach Ollama at "
                f"{self.base_url}. Start it with `ollama serve` or enable the Ollama service."
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned a response that was not valid JSON.") from exc

        names: list[str] = []
        for item in payload.get("models", []):
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
                if isinstance(name, str):
                    names.append(name)
        return sorted(names)

    def ensure_model_available(self, model: str | None = None) -> None:
        selected_model = model or self.model
        available_models = self.list_models()
        if selected_model not in available_models:
            model_list = ", ".join(available_models) if available_models else "none"
            raise OllamaModelNotFoundError(
                f"Model {selected_model!r} is not available in Ollama. "
                f"Run `ollama pull {selected_model}`. Available models: {model_list}."
            )

    def chat(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        selected_model = model or self.model
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
        }
        started_at = perf_counter()

        try:
            response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            self._record_llm_error(selected_model, started_at, str(exc))
            raise OllamaConnectionError(
                "Could not send the chat request to Ollama. "
                f"Check that Ollama is running at {self.base_url}."
            ) from exc

        if response.status_code == 404 or "not found" in response.text.lower():
            self._record_llm_error(selected_model, started_at, response.text[:300])
            raise OllamaModelNotFoundError(
                f"Ollama could not find model {selected_model!r}. "
                f"Run `ollama pull {selected_model}` and try again."
            )

        if not response.ok:
            details = response.text.strip()[:500]
            self._record_llm_error(selected_model, started_at, details)
            raise OllamaError(
                f"Ollama returned HTTP {response.status_code}. "
                f"Response details: {details or 'no details'}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            self._record_llm_error(selected_model, started_at, "invalid json")
            raise OllamaError("Ollama returned a chat response that was not valid JSON.") from exc

        message = data.get("message")
        if not isinstance(message, dict):
            self._record_llm_error(selected_model, started_at, "missing assistant message")
            raise OllamaError("Ollama response did not include an assistant message.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            self._record_llm_error(selected_model, started_at, "empty assistant response")
            raise OllamaError("Ollama returned an empty assistant response.")

        answer = content.strip()
        self._record_llm_success(
            selected_model,
            messages,
            answer,
            data,
            started_at,
        )
        return answer

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> Iterator[str]:
        selected_model = model or self.model
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "stream": True,
        }
        started_at = perf_counter()
        chunks: list[str] = []

        try:
            with self.session.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
                stream=True,
            ) as response:
                if response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Ollama could not find model {selected_model!r}. "
                        f"Run `ollama pull {selected_model}` and try again."
                    )

                if not response.ok:
                    details = response.text.strip()[:500]
                    raise OllamaError(
                        f"Ollama returned HTTP {response.status_code}. "
                        f"Response details: {details or 'no details'}"
                    )

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError("Ollama returned an invalid streaming JSON line.") from exc

                    error = data.get("error")
                    if isinstance(error, str) and error:
                        if "not found" in error.lower():
                            raise OllamaModelNotFoundError(
                                f"Ollama could not find model {selected_model!r}. "
                                f"Run `ollama pull {selected_model}` and try again."
                            )
                        raise OllamaError(f"Ollama streaming error: {error}")

                    message = data.get("message")
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            chunks.append(content)
                            yield content

                    if data.get("done"):
                        self._record_llm_success(
                            selected_model,
                            messages,
                            "".join(chunks).strip(),
                            data,
                            started_at,
                        )
                        break
        except requests.RequestException as exc:
            self._record_llm_error(selected_model, started_at, str(exc))
            raise OllamaConnectionError(
                "Could not stream the chat response from Ollama. "
                f"Check that Ollama is running at {self.base_url}."
            ) from exc

    def _record_llm_success(
        self,
        model: str,
        messages: list[dict[str, str]],
        response_text: str,
        data: dict[str, Any],
        started_at: float,
    ) -> None:
        latency_seconds = perf_counter() - started_at
        prompt_tokens = _optional_int(data.get("prompt_eval_count"))
        completion_tokens = _optional_int(data.get("eval_count"))
        token_source = "ollama" if prompt_tokens is not None or completion_tokens is not None else "estimated"
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens_from_messages(messages)
        if completion_tokens is None:
            completion_tokens = estimate_tokens_from_text(response_text)

        if self.config.metrics_enabled and self.config.prometheus_enabled:
            observe_llm_call(
                model=model,
                provider="ollama",
                status="success",
                latency_seconds=latency_seconds,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                token_source=token_source,
            )

        ollama_metadata = _ollama_metadata(data)
        langfuse_metadata = {
            **ollama_metadata,
            "provider": "ollama",
            "status": "success",
            "latency_ms": round(latency_seconds * 1000, 2),
            "token_source": token_source,
        }
        get_langfuse_tracer().log_generation(
            current_trace(),
            name="ollama_chat",
            model=model,
            messages=messages,
            response_text=response_text,
            metadata=langfuse_metadata,
            usage={"input": prompt_tokens, "output": completion_tokens},
        )
        log_event(
            logger,
            "llm_call",
            config=self.config,
            model=model,
            provider="ollama",
            status="success",
            latency_ms=round(latency_seconds * 1000, 2),
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            token_source=token_source,
            **{f"ollama_{key}": value for key, value in ollama_metadata.items()},
        )

    def _record_llm_error(self, model: str, started_at: float, error_message: str) -> None:
        latency_seconds = perf_counter() - started_at
        if self.config.metrics_enabled and self.config.prometheus_enabled:
            observe_llm_call(
                model=model,
                provider="ollama",
                status="error",
                latency_seconds=latency_seconds,
            )
        log_event(
            logger,
            "llm_call",
            config=self.config,
            model=model,
            provider="ollama",
            status="error",
            latency_ms=round(latency_seconds * 1000, 2),
            error_message=error_message,
        )


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _ollama_metadata(data: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ]
    return {key: data[key] for key in keys if key in data}

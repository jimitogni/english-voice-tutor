from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests

from app.config import AppConfig, load_config


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

        try:
            response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                "Could not send the chat request to Ollama. "
                f"Check that Ollama is running at {self.base_url}."
            ) from exc

        if response.status_code == 404 or "not found" in response.text.lower():
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

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned a chat response that was not valid JSON.") from exc

        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaError("Ollama response did not include an assistant message.")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("Ollama returned an empty assistant response.")

        return content.strip()

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
                            yield content

                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            raise OllamaConnectionError(
                "Could not stream the chat response from Ollama. "
                f"Check that Ollama is running at {self.base_url}."
            ) from exc

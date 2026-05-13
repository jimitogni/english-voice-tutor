from __future__ import annotations

from time import perf_counter
from typing import Any

import requests

from app.config import AppConfig, load_config
from app.observability.metrics import observe_embedding_call
from app.rag.models import RagError


class EmbeddingError(RagError):
    """Raised when the embedding model cannot produce usable vectors."""


class OllamaEmbeddingClient:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.base_url = self.config.ollama_base_url.rstrip("/")
        self.model = self.config.rag_embedding_model
        self.timeout_seconds = self.config.rag_request_timeout_seconds
        self.session = requests.Session()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text.strip() for text in texts if text.strip()]
        if not clean_texts:
            return []

        started_at = perf_counter()
        try:
            embeddings = self._embed(clean_texts)
        except Exception as exc:
            latency_seconds = perf_counter() - started_at
            if self.config.metrics_enabled and self.config.prometheus_enabled:
                observe_embedding_call(
                    model=self.model,
                    provider="ollama",
                    status="error",
                    latency_seconds=latency_seconds,
                    input_count=len(clean_texts),
                )
            if isinstance(exc, EmbeddingError):
                raise
            raise EmbeddingError(f"Could not generate embeddings with Ollama model {self.model!r}.") from exc

        latency_seconds = perf_counter() - started_at
        if self.config.metrics_enabled and self.config.prometheus_enabled:
            observe_embedding_call(
                model=self.model,
                provider="ollama",
                status="success",
                latency_seconds=latency_seconds,
                input_count=len(clean_texts),
            )
        return embeddings

    def _embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise EmbeddingError(
                f"Could not reach Ollama embeddings endpoint at {self.base_url}."
            ) from exc

        if response.status_code == 404 or "not found" in response.text.lower():
            raise EmbeddingError(
                f"Ollama embedding model {self.model!r} is not available. "
                f"Run `ollama pull {self.model}` and index knowledge again."
            )

        if not response.ok:
            details = response.text.strip()[:500]
            raise EmbeddingError(
                f"Ollama embeddings returned HTTP {response.status_code}: "
                f"{details or 'no details'}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise EmbeddingError("Ollama embeddings response was not valid JSON.") from exc

        embeddings = _parse_embeddings(data, expected_count=len(texts))
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs."
            )
        return embeddings


def _parse_embeddings(data: dict[str, Any], expected_count: int) -> list[list[float]]:
    raw_embeddings = data.get("embeddings")
    if isinstance(raw_embeddings, list):
        return [_coerce_embedding(item) for item in raw_embeddings]

    raw_embedding = data.get("embedding")
    if expected_count == 1 and isinstance(raw_embedding, list):
        return [_coerce_embedding(raw_embedding)]

    raise EmbeddingError("Ollama embeddings response did not include embeddings.")


def _coerce_embedding(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise EmbeddingError("Ollama returned an empty or invalid embedding.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("Ollama returned a non-numeric embedding.") from exc

from __future__ import annotations

from typing import Any

import requests

from app.config import AppConfig, load_config
from app.rag.models import KnowledgeChunk, RagError, RagSource


class QdrantStoreError(RagError):
    """Raised when Qdrant cannot read or write vector data."""


class QdrantVectorStore:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.base_url = self.config.qdrant_url.rstrip("/")
        self.collection = self.config.qdrant_collection
        self.timeout_seconds = self.config.rag_request_timeout_seconds
        self.session = requests.Session()

    def collection_exists(self) -> bool:
        try:
            response = self.session.get(
                self._url(f"/collections/{self.collection}"),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise QdrantStoreError(f"Could not reach Qdrant at {self.base_url}.") from exc

        if response.status_code == 404:
            return False
        if not response.ok:
            details = response.text.strip()[:500]
            raise QdrantStoreError(
                f"Qdrant collection check returned HTTP {response.status_code}: "
                f"{details or 'no details'}"
            )
        return True

    def ensure_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise QdrantStoreError("Vector size must be greater than zero.")
        if self.collection_exists():
            return

        payload = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        }
        try:
            response = self.session.put(
                self._url(f"/collections/{self.collection}"),
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise QdrantStoreError(f"Could not create Qdrant collection {self.collection!r}.") from exc

        if not response.ok:
            details = response.text.strip()[:500]
            raise QdrantStoreError(
                f"Qdrant collection creation returned HTTP {response.status_code}: "
                f"{details or 'no details'}"
            )

    def delete_collection(self) -> bool:
        try:
            response = self.session.delete(
                self._url(f"/collections/{self.collection}"),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise QdrantStoreError(f"Could not delete Qdrant collection {self.collection!r}.") from exc

        if response.status_code == 404:
            return False
        if not response.ok:
            details = response.text.strip()[:500]
            raise QdrantStoreError(
                f"Qdrant collection delete returned HTTP {response.status_code}: "
                f"{details or 'no details'}"
            )
        return True

    def upsert(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise QdrantStoreError("Chunk and embedding counts do not match.")
        if not chunks:
            return 0

        points = [
            {
                "id": chunk.id,
                "vector": embedding,
                "payload": {
                    "title": chunk.title,
                    "source": chunk.source,
                    "content": chunk.content,
                    **chunk.metadata,
                },
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        try:
            response = self.session.put(
                self._url(f"/collections/{self.collection}/points"),
                params={"wait": "true"},
                json={"points": points},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise QdrantStoreError(f"Could not upsert points into Qdrant {self.collection!r}.") from exc

        if not response.ok:
            details = response.text.strip()[:500]
            raise QdrantStoreError(
                f"Qdrant upsert returned HTTP {response.status_code}: {details or 'no details'}"
            )
        return len(points)

    def query(
        self,
        embedding: list[float],
        *,
        limit: int,
        score_threshold: float | None,
    ) -> list[RagSource]:
        if not embedding:
            return []
        query_body: dict[str, Any] = {
            "query": embedding,
            "limit": limit,
            "with_payload": True,
        }
        if score_threshold is not None:
            query_body["score_threshold"] = score_threshold

        response = self._post_query("/points/query", query_body)
        if response.status_code == 404:
            legacy_body: dict[str, Any] = {
                "vector": embedding,
                "limit": limit,
                "with_payload": True,
            }
            if score_threshold is not None:
                legacy_body["score_threshold"] = score_threshold
            response = self._post_query("/points/search", legacy_body)

        if response.status_code == 404:
            return []
        if not response.ok:
            details = response.text.strip()[:500]
            raise QdrantStoreError(
                f"Qdrant query returned HTTP {response.status_code}: {details or 'no details'}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise QdrantStoreError("Qdrant query response was not valid JSON.") from exc

        return _parse_query_response(data)

    def _post_query(self, endpoint: str, body: dict[str, Any]) -> requests.Response:
        try:
            return self.session.post(
                self._url(f"/collections/{self.collection}{endpoint}"),
                json=body,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise QdrantStoreError(f"Could not query Qdrant collection {self.collection!r}.") from exc

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"


def _parse_query_response(data: dict[str, Any]) -> list[RagSource]:
    result = data.get("result")
    if isinstance(result, dict) and isinstance(result.get("points"), list):
        raw_points = result["points"]
    elif isinstance(result, list):
        raw_points = result
    else:
        raw_points = []

    sources: list[RagSource] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            continue
        payload = raw_point.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        content = payload.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        title = payload.get("title")
        source = payload.get("source")
        score = _optional_float(raw_point.get("score"))
        sources.append(
            RagSource(
                id=str(raw_point.get("id") or ""),
                title=title if isinstance(title, str) and title else "Untitled",
                source=source if isinstance(source, str) and source else "unknown",
                content=content.strip(),
                score=score,
                metadata={key: value for key, value in payload.items() if key != "content"},
            )
        )
    return sources


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

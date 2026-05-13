from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class RagError(RuntimeError):
    """Raised when a RAG dependency cannot complete a requested operation."""


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    title: str
    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RagSource:
    id: str
    title: str
    source: str
    content: str
    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalContext:
    sources: list[RagSource]
    vector_db: str = "qdrant"
    error: str | None = None
    latency_seconds: float = 0.0

    @property
    def count(self) -> int:
        return len(self.sources)

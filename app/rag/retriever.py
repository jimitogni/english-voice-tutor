from __future__ import annotations

from time import perf_counter

from app.config import AppConfig, load_config
from app.observability.metrics import observe_rag_retrieval
from app.rag.embeddings import OllamaEmbeddingClient
from app.rag.models import RagError, RagSource, RetrievalContext
from app.rag.qdrant_store import QdrantVectorStore


class RagRetriever:
    def __init__(
        self,
        config: AppConfig | None = None,
        embeddings: OllamaEmbeddingClient | None = None,
        store: QdrantVectorStore | None = None,
    ) -> None:
        self.config = config or load_config()
        self.embeddings = embeddings or OllamaEmbeddingClient(self.config)
        self.store = store or QdrantVectorStore(self.config)

    def retrieve(self, query: str) -> RetrievalContext:
        if not self.config.rag_enabled:
            return RetrievalContext(sources=[], vector_db=self.config.rag_vector_db)

        started_at = perf_counter()
        status = "success"
        sources: list[RagSource] = []
        error: str | None = None

        try:
            if not self.store.collection_exists():
                status = "empty"
            else:
                query_embedding = self.embeddings.embed_texts([query])
                if query_embedding:
                    sources = self.store.query(
                        query_embedding[0],
                        limit=self.config.rag_top_k,
                        score_threshold=self.config.rag_score_threshold,
                    )
                if not sources:
                    status = "empty"
            if status == "success" and not sources:
                status = "empty"
        except RagError as exc:
            status = "error"
            error = str(exc)

        latency_seconds = perf_counter() - started_at
        if self.config.metrics_enabled and self.config.prometheus_enabled:
            observe_rag_retrieval(
                vector_db=self.config.rag_vector_db,
                status=status,
                latency_seconds=latency_seconds,
                result_count=len(sources),
            )
        return RetrievalContext(
            sources=sources,
            vector_db=self.config.rag_vector_db,
            error=error,
            latency_seconds=latency_seconds,
        )


def format_retrieval_context(sources: list[RagSource], *, max_chars: int) -> str:
    if not sources:
        return ""

    parts = [
        "Relevant local reference material was retrieved for this turn.",
        "Use it only when it helps answer or coach the user.",
        "Do not mention source names unless the user asks where the information came from.",
    ]
    used_chars = 0
    for index, source in enumerate(sources, start=1):
        remaining_chars = max_chars - used_chars
        if remaining_chars <= 0:
            break
        content = source.content.strip()
        if len(content) > remaining_chars:
            content = content[:remaining_chars].rsplit(" ", 1)[0].strip()
        if not content:
            break
        score = f"{source.score:.3f}" if source.score is not None else "unknown"
        parts.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"Title: {source.title}",
                    f"Path: {source.source}",
                    f"Score: {score}",
                    f"Content: {content}",
                ]
            )
        )
        used_chars += len(content)

    return "\n\n".join(parts)

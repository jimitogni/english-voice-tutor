from __future__ import annotations

from app.config import load_config
from app.evaluation.evidently_llm_eval import parse_source_scores, parse_sources
from app.rag.documents import chunk_text_by_chars
from app.rag.models import RagSource
from app.rag.retriever import format_retrieval_context


def test_rag_config_loads_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embeddinggemma")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    config = load_config()

    assert config.rag_enabled is True
    assert config.rag_embedding_model == "embeddinggemma"
    assert config.qdrant_url == "http://qdrant:6333"


def test_chunk_text_by_chars_keeps_content_and_overlap() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."

    chunks = chunk_text_by_chars(text, chunk_chars=32, overlap=8)

    assert len(chunks) >= 2
    assert chunks[0].startswith("First sentence.")
    assert chunks[-1].endswith("Fourth sentence.")


def test_format_retrieval_context_limits_content() -> None:
    source = RagSource(
        id="source-id",
        title="Interview Notes",
        source="interview.md",
        content="Monitoring deployment production metrics " * 20,
        score=0.91,
    )

    context = format_retrieval_context([source], max_chars=80)

    assert "Interview Notes" in context
    assert "interview.md" in context
    assert "0.910" in context
    assert len(context) < 600


def test_eval_source_parsing_handles_api_shape() -> None:
    sources = [
        {"title": "Grammar", "source": "grammar.md", "score": 0.8},
        {"title": "Fallback", "score": "0.6"},
        {"source": "", "score": None},
    ]

    assert parse_sources(sources) == ["grammar.md", "Fallback"]
    assert parse_source_scores(sources) == [0.8, 0.6]

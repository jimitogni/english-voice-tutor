from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from app.config import AppConfig, load_config
from app.rag.models import KnowledgeChunk
from app.utils import ensure_directory


SUPPORTED_KNOWLEDGE_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".jsonl", ".csv"}


def load_knowledge_chunks(config: AppConfig | None = None) -> list[KnowledgeChunk]:
    selected_config = config or load_config()
    ensure_directory(selected_config.knowledge_dir)

    chunks: list[KnowledgeChunk] = []
    for path in _iter_knowledge_files(selected_config.knowledge_dir):
        text = _read_text(path)
        if not text.strip():
            continue
        relative_path = path.relative_to(selected_config.knowledge_dir)
        title = _title_from_path(relative_path)
        for index, chunk_text in enumerate(
            chunk_text_by_chars(
                text,
                chunk_chars=selected_config.rag_chunk_chars,
                overlap=selected_config.rag_chunk_overlap,
            )
        ):
            chunk_id = _chunk_id(str(relative_path), index, chunk_text)
            chunks.append(
                KnowledgeChunk(
                    id=chunk_id,
                    title=title,
                    source=str(relative_path),
                    content=chunk_text,
                    metadata={
                        "source_type": "knowledge_file",
                        "path": str(relative_path),
                        "chunk_index": index,
                        "content_sha256": hashlib.sha256(
                            chunk_text.encode("utf-8")
                        ).hexdigest(),
                    },
                )
            )
    return chunks


def chunk_text_by_chars(text: str, *, chunk_chars: int, overlap: int) -> list[str]:
    clean_text = _normalize_text(text)
    if not clean_text:
        return []

    chunk_size = max(chunk_chars, 1)
    overlap_size = min(max(overlap, 0), max(chunk_size - 1, 0))
    chunks: list[str] = []
    start = 0

    while start < len(clean_text):
        hard_end = min(start + chunk_size, len(clean_text))
        end = hard_end
        if hard_end < len(clean_text):
            soft_end = _soft_break(clean_text, start, hard_end)
            if soft_end > start:
                end = soft_end

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(clean_text):
            break
        start = max(end - overlap_size, start + 1)

    return chunks


def _iter_knowledge_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_KNOWLEDGE_SUFFIXES:
            continue
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        files.append(path)
    return files


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def _soft_break(text: str, start: int, hard_end: int) -> int:
    candidates = [
        text.rfind("\n\n", start, hard_end),
        text.rfind(". ", start, hard_end),
        text.rfind("? ", start, hard_end),
        text.rfind("! ", start, hard_end),
        text.rfind("\n", start, hard_end),
        text.rfind(" ", start, hard_end),
    ]
    minimum = start + max((hard_end - start) // 2, 1)
    for candidate in candidates:
        if candidate >= minimum:
            return candidate + 1
    return hard_end


def _title_from_path(relative_path: Path) -> str:
    return relative_path.stem.replace("_", " ").replace("-", " ").strip().title() or str(relative_path)


def _chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}:{index}:{content}".encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"english-voice-tutor-rag:{digest}"))

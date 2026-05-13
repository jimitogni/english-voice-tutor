#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.rag.documents import load_knowledge_chunks
from app.rag.embeddings import OllamaEmbeddingClient
from app.rag.models import KnowledgeChunk, RagError
from app.rag.qdrant_store import QdrantVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index local knowledge files into Qdrant for RAG.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the configured Qdrant collection before indexing.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of chunks to embed and upsert per batch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        print("--batch-size must be greater than zero.", file=sys.stderr)
        return 2

    config = load_config()
    chunks = load_knowledge_chunks(config)
    if not chunks:
        print(f"No knowledge files found in {config.knowledge_dir}.")
        print("Add .md, .txt, .json, .jsonl, or .csv files and run this script again.")
        return 0

    store = QdrantVectorStore(config)
    embeddings = OllamaEmbeddingClient(config)

    try:
        if args.reset and store.delete_collection():
            print(f"Deleted existing Qdrant collection: {config.qdrant_collection}")

        indexed = index_chunks(
            chunks=chunks,
            embeddings=embeddings,
            store=store,
            batch_size=args.batch_size,
        )
    except RagError as exc:
        print(f"RAG indexing failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Indexed {indexed} chunks into Qdrant collection "
        f"{config.qdrant_collection!r} from {config.knowledge_dir}."
    )
    return 0


def index_chunks(
    *,
    chunks: list[KnowledgeChunk],
    embeddings: OllamaEmbeddingClient,
    store: QdrantVectorStore,
    batch_size: int,
) -> int:
    indexed = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        vectors = embeddings.embed_texts([chunk.content for chunk in batch])
        if vectors and indexed == 0:
            store.ensure_collection(len(vectors[0]))
        indexed += store.upsert(batch, vectors)
        print(f"Indexed {indexed}/{len(chunks)} chunks...")
    return indexed


if __name__ == "__main__":
    raise SystemExit(main())

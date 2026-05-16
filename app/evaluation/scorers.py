from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Protocol

from app.config import AppConfig

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerBackend:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


def build_embedding_backend(config: AppConfig) -> EmbeddingBackend:
    backend = config.evaluation_embedding_backend.strip().lower()
    if backend == "sentence_transformers":
        return _cached_sentence_transformer_backend(config.evaluation_embedding_model)
    from app.rag.embeddings import OllamaEmbeddingClient

    return OllamaEmbeddingClient(config)


def score_reference_metrics(candidate: str, reference: str) -> dict[str, float]:
    if not candidate.strip() or not reference.strip():
        return {
            "bleu_score": 0.0,
            "rouge1_f1": 0.0,
            "rouge_l_f1": 0.0,
            "grammar_correction_quality": 0.0,
        }

    # BLEU and ROUGE are useful only when a reference answer exists. They help compare
    # overlap with an expected answer, but they are not sufficient for open-ended LLM
    # evaluation because valid responses can use different wording while still being correct.
    bleu_score = _bleu(candidate, reference)
    rouge1_f1 = _rouge_n_f1(candidate, reference, n=1)
    rouge_l_f1 = _rouge_l_f1(candidate, reference)
    grammar_quality = round(
        (normalized_exact_match(candidate, reference) + normalized_edit_similarity(candidate, reference))
        / 2.0,
        4,
    )
    return {
        "bleu_score": bleu_score,
        "rouge1_f1": rouge1_f1,
        "rouge_l_f1": rouge_l_f1,
        "grammar_correction_quality": grammar_quality,
    }


def semantic_similarity(
    candidate: str,
    reference: str,
    *,
    backend: EmbeddingBackend | None,
) -> float | None:
    if not candidate.strip() or not reference.strip() or backend is None:
        return None
    try:
        vectors = backend.embed_texts([candidate, reference])
    except Exception:
        return None
    if len(vectors) != 2:
        return None
    return round(_cosine_similarity(vectors[0], vectors[1]), 4)


def hallucination_score(candidate: str, reference_context: str, *, semantic_score: float | None) -> float | None:
    if not candidate.strip() or not reference_context.strip():
        return None
    candidate_terms = set(_content_tokens(candidate))
    if not candidate_terms:
        return 1.0
    context_terms = set(_content_tokens(reference_context))
    lexical_support = len(candidate_terms & context_terms) / len(candidate_terms)
    if semantic_score is None:
        return round(lexical_support, 4)
    return round((lexical_support + semantic_score) / 2.0, 4)


def normalized_exact_match(candidate: str, reference: str) -> float:
    return 1.0 if _normalize_text(candidate) == _normalize_text(reference) else 0.0


def normalized_edit_similarity(candidate: str, reference: str) -> float:
    return round(SequenceMatcher(None, _normalize_text(candidate), _normalize_text(reference)).ratio(), 4)


def response_lengths(text: str) -> tuple[int, int]:
    stripped = text.strip()
    if not stripped:
        return 0, 0
    return len(stripped), len(stripped.split())


@lru_cache(maxsize=1)
def sentence_transformers_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=2)
def _cached_sentence_transformer_backend(model_name: str) -> SentenceTransformerBackend:
    return SentenceTransformerBackend(model_name)


def _bleu(candidate: str, reference: str) -> float:
    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, 5):
        candidate_ngrams = Counter(_ngrams(candidate_tokens, n))
        reference_ngrams = Counter(_ngrams(reference_tokens, n))
        total = sum(candidate_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, reference_ngrams[gram]) for gram, count in candidate_ngrams.items())
        precisions.append((overlap + 1) / (total + 1))

    brevity_penalty = 1.0
    if len(candidate_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(1 - (len(reference_tokens) / max(len(candidate_tokens), 1)))

    score = brevity_penalty * math.exp(sum(math.log(max(p, 1e-9)) for p in precisions) / 4)
    return round(score, 4)


def _rouge_n_f1(candidate: str, reference: str, *, n: int) -> float:
    candidate_ngrams = Counter(_ngrams(_tokens(candidate), n))
    reference_ngrams = Counter(_ngrams(_tokens(reference), n))
    if not candidate_ngrams or not reference_ngrams:
        return 0.0
    overlap = sum(min(count, reference_ngrams[gram]) for gram, count in candidate_ngrams.items())
    precision = overlap / sum(candidate_ngrams.values())
    recall = overlap / sum(reference_ngrams.values())
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def _rouge_l_f1(candidate: str, reference: str) -> float:
    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    lcs = _longest_common_subsequence(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall) / (precision + recall), 4)


def _longest_common_subsequence(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(min(dot / (left_norm * right_norm), 1.0), -1.0)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[index : index + n]) for index in range(0, max(len(tokens) - n + 1, 0))]


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(_normalize_text(text))


def _content_tokens(text: str) -> list[str]:
    return [token for token in _tokens(text) if token not in STOPWORDS]


def _normalize_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))

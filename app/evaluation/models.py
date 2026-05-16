from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class EvalExample:
    id: str
    input: str
    expected_output: str | None = None
    task_type: str = "general"
    tags: list[str] = field(default_factory=list)
    reference_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_tool_calls: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalExample":
        example_id = str(payload.get("id") or payload.get("request_id") or "").strip()
        if not example_id:
            input_text = str(payload.get("input") or payload.get("question") or "").strip()
            example_id = f"example-{abs(hash(input_text))}"

        input_text = str(payload.get("input") or payload.get("question") or "").strip()
        expected_output = payload.get("expected_output", payload.get("expected_answer"))
        reference_context = payload.get("reference_context")
        expected_tool_calls = payload.get("expected_tool_calls")
        raw_tags = payload.get("tags")
        if raw_tags:
            tags = raw_tags
        elif payload.get("category"):
            tags = [payload.get("category")]
        else:
            tags = []
        metadata = dict(payload.get("metadata") or {})
        if "difficulty" in payload:
            metadata.setdefault("difficulty", payload["difficulty"])
        if "expected_context_keywords" in payload:
            metadata.setdefault("expected_context_keywords", payload["expected_context_keywords"])

        return cls(
            id=example_id,
            input=input_text,
            expected_output=str(expected_output).strip() if expected_output is not None else None,
            task_type=str(payload.get("task_type") or payload.get("category") or "general"),
            tags=_normalize_tags(tags),
            reference_context=str(reference_context).strip() if reference_context else None,
            metadata=metadata,
            expected_tool_calls=_coerce_optional_int(expected_tool_calls),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    status: str
    latency_ms: float | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LlmUsageRecord:
    model: str
    provider: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    token_source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationMetrics:
    bleu_score: float | None = None
    rouge1_f1: float | None = None
    rouge_l_f1: float | None = None
    semantic_similarity: float | None = None
    response_length_chars: int = 0
    response_length_words: int = 0
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    error_rate: float = 0.0
    hallucination_score: float | None = None
    factuality_score: float | None = None
    grammar_correction_quality: float | None = None
    user_feedback_score: float | None = None
    task_success_rate: float | None = None
    tool_call_success_rate: float | None = None
    tool_call_error_rate: float | None = None
    tool_calls_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationRecord:
    id: str
    request_id: str
    session_id: str | None
    timestamp: str
    task_type: str
    tags: list[str]
    input_text: str
    output_text: str
    expected_output: str | None
    reference_context: str | None
    metrics: EvaluationMetrics
    model_name: str
    provider: str
    llm_usage: LlmUsageRecord | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    workflow_status: str = "success"
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(self.metrics.to_dict())
        payload["metrics"] = self.metrics.to_dict()
        payload["llm_usage"] = self.llm_usage.to_dict() if self.llm_usage is not None else None
        payload["tool_calls"] = [tool_call.to_dict() for tool_call in self.tool_calls]
        return payload


@dataclass(frozen=True)
class ExperimentRunSummary:
    run_id: str
    created_at: str
    dataset_path: str
    dataset_size: int
    model_name: str | None
    git_commit: str | None
    records_path: str
    summary_path: str
    averages: dict[str, float]
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import uuid4

from app.config import AppConfig, load_config
from app.evaluation.models import (
    EvalExample,
    EvaluationMetrics,
    EvaluationRecord,
    ExperimentRunSummary,
    LlmUsageRecord,
    ToolCallRecord,
    utc_now_iso,
)
from app.evaluation.scorers import build_embedding_backend, hallucination_score, response_lengths, score_reference_metrics, semantic_similarity
from app.observability.langfuse_client import get_langfuse_tracer


class EvaluationService:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.embedding_backend = None
        if self.config.evaluation_enabled:
            self.embedding_backend = self._build_embedding_backend()

    def evaluate_interaction(
        self,
        *,
        request_id: str,
        session_id: str | None,
        input_text: str,
        output_text: str,
        model_name: str,
        provider: str,
        task_type: str = "general",
        tags: list[str] | None = None,
        expected_output: str | None = None,
        reference_context: str | None = None,
        latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        token_source: str | None = None,
        error_message: str | None = None,
        tool_calls: list[ToolCallRecord] | None = None,
        task_success: bool | None = None,
        user_feedback_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationRecord:
        usage = self._build_usage(
            model_name=model_name,
            provider=provider,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            token_source=token_source,
        )
        metrics = self._build_metrics(
            output_text=output_text,
            expected_output=expected_output,
            reference_context=reference_context,
            latency_ms=latency_ms,
            usage=usage,
            error_message=error_message,
            tool_calls=tool_calls or [],
            task_success=task_success,
            user_feedback_score=user_feedback_score,
        )
        return EvaluationRecord(
            id=request_id,
            request_id=request_id,
            session_id=session_id,
            timestamp=utc_now_iso(),
            task_type=task_type,
            tags=tags or [],
            input_text=input_text,
            output_text=output_text,
            expected_output=expected_output,
            reference_context=reference_context,
            metrics=metrics,
            model_name=model_name,
            provider=provider,
            llm_usage=usage,
            tool_calls=tool_calls or [],
            workflow_status="error" if error_message else "success",
            error_message=error_message,
            metadata=metadata or {},
        )

    def persist_interaction(self, record: EvaluationRecord) -> Path | None:
        if not self.config.evaluation_enabled:
            return None
        interactions_dir = self.config.evaluation_data_dir / "interactions"
        interactions_dir.mkdir(parents=True, exist_ok=True)
        path = interactions_dir / f"{record.request_id}.json"
        path.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self._score_langfuse(record)
        return path

    def persist_feedback(self, *, request_id: str, score: float, comment: str | None = None) -> Path | None:
        if not self.config.evaluation_enabled:
            return None
        path = self.config.evaluation_data_dir / "interactions" / f"{request_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["user_feedback_score"] = score
        payload["feedback_comment"] = comment
        metrics = payload.get("metrics") or {}
        metrics["user_feedback_score"] = score
        payload["metrics"] = metrics
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tracer = get_langfuse_tracer()
        tracer.score(trace_id=request_id, name="user_feedback_score", value=score, comment=comment)
        tracer.flush()
        return path

    def load_dataset(self, path: Path) -> list[EvalExample]:
        examples: list[EvalExample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                clean_line = line.strip()
                if clean_line:
                    examples.append(EvalExample.from_dict(json.loads(clean_line)))
        return examples

    def save_run_results(
        self,
        *,
        dataset_path: Path,
        records: list[EvaluationRecord],
        output_dir: Path,
        model_name: str | None,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid4().hex[:12]
        records_path = output_dir / f"eval_results_{run_id}.jsonl"
        with records_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        summary = ExperimentRunSummary(
            run_id=run_id,
            created_at=utc_now_iso(),
            dataset_path=str(dataset_path),
            dataset_size=len(records),
            model_name=model_name,
            git_commit=self._git_commit(),
            records_path=str(records_path),
            summary_path="",
            averages=self._average_metrics(records),
            counts=self._count_metrics(records),
        )
        summary_path = output_dir / f"eval_summary_{run_id}.json"
        summary = ExperimentRunSummary(
            **{**summary.to_dict(), "summary_path": str(summary_path)}
        )
        summary_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return records_path, summary_path

    def _build_embedding_backend(self):
        try:
            return build_embedding_backend(self.config)
        except (ImportError, OSError, RuntimeError):
            return None

    def _build_usage(
        self,
        *,
        model_name: str,
        provider: str,
        latency_ms: float,
        input_tokens: int | None,
        output_tokens: int | None,
        token_source: str | None,
    ) -> LlmUsageRecord:
        total_tokens = None
        if input_tokens is not None or output_tokens is not None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        return LlmUsageRecord(
            model=model_name,
            provider=provider,
            latency_ms=round(latency_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self._estimate_cost(input_tokens, output_tokens),
            token_source=token_source,
        )

    def _build_metrics(
        self,
        *,
        output_text: str,
        expected_output: str | None,
        reference_context: str | None,
        latency_ms: float,
        usage: LlmUsageRecord,
        error_message: str | None,
        tool_calls: list[ToolCallRecord],
        task_success: bool | None,
        user_feedback_score: float | None,
    ) -> EvaluationMetrics:
        response_length_chars, response_length_words = response_lengths(output_text)
        bleu_score = None
        rouge1_f1 = None
        rouge_l_f1 = None
        grammar_quality = None
        semantic_score = None
        factuality = None

        if expected_output:
            reference_scores = score_reference_metrics(output_text, expected_output)
            bleu_score = reference_scores["bleu_score"]
            rouge1_f1 = reference_scores["rouge1_f1"]
            rouge_l_f1 = reference_scores["rouge_l_f1"]
            grammar_quality = reference_scores["grammar_correction_quality"]
            semantic_score = semantic_similarity(
                output_text,
                expected_output,
                backend=self.embedding_backend,
            )

        if reference_context:
            context_semantic_score = semantic_similarity(
                output_text,
                reference_context,
                backend=self.embedding_backend,
            )
            factuality = hallucination_score(
                output_text,
                reference_context,
                semantic_score=context_semantic_score,
            )

        tool_successes = sum(1 for tool_call in tool_calls if tool_call.status == "success")
        tool_errors = sum(1 for tool_call in tool_calls if tool_call.status == "error")
        tool_call_success_rate = None
        tool_call_error_rate = None
        if tool_calls:
            tool_call_success_rate = round(tool_successes / len(tool_calls), 4)
            tool_call_error_rate = round(tool_errors / len(tool_calls), 4)

        effective_task_success = task_success
        if effective_task_success is None:
            effective_task_success = error_message is None

        return EvaluationMetrics(
            bleu_score=bleu_score,
            rouge1_f1=rouge1_f1,
            rouge_l_f1=rouge_l_f1,
            semantic_similarity=semantic_score,
            response_length_chars=response_length_chars,
            response_length_words=response_length_words,
            latency_ms=round(latency_ms, 2),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            error_rate=1.0 if error_message else 0.0,
            hallucination_score=factuality,
            factuality_score=factuality,
            grammar_correction_quality=grammar_quality,
            user_feedback_score=user_feedback_score,
            task_success_rate=float(effective_task_success),
            tool_call_success_rate=tool_call_success_rate,
            tool_call_error_rate=tool_call_error_rate,
            tool_calls_count=len(tool_calls),
        )

    def _estimate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        if input_tokens is None and output_tokens is None:
            return None
        input_cost = ((input_tokens or 0) / 1_000_000) * self.config.evaluation_cost_input_per_million
        output_cost = ((output_tokens or 0) / 1_000_000) * self.config.evaluation_cost_output_per_million
        return round(input_cost + output_cost, 8)

    def _average_metrics(self, records: list[EvaluationRecord]) -> dict[str, float]:
        return {
            "bleu_score": _mean(record.metrics.bleu_score for record in records),
            "rouge1_f1": _mean(record.metrics.rouge1_f1 for record in records),
            "rouge_l_f1": _mean(record.metrics.rouge_l_f1 for record in records),
            "semantic_similarity": _mean(record.metrics.semantic_similarity for record in records),
            "latency_ms": _mean(record.metrics.latency_ms for record in records),
            "error_rate": _mean(record.metrics.error_rate for record in records),
            "factuality_score": _mean(record.metrics.factuality_score for record in records),
            "grammar_correction_quality": _mean(record.metrics.grammar_correction_quality for record in records),
            "task_success_rate": _mean(record.metrics.task_success_rate for record in records),
            "tool_call_success_rate": _mean(record.metrics.tool_call_success_rate for record in records),
            "tool_call_error_rate": _mean(record.metrics.tool_call_error_rate for record in records),
            "tool_calls_count": _mean(record.metrics.tool_calls_count for record in records),
        }

    def _count_metrics(self, records: list[EvaluationRecord]) -> dict[str, int]:
        return {
            "records": len(records),
            "errors": sum(1 for record in records if record.error_message),
            "tool_calls": sum(record.metrics.tool_calls_count for record in records),
            "tool_errors": sum(
                1 for record in records for tool_call in record.tool_calls if tool_call.status == "error"
            ),
        }

    def _score_langfuse(self, record: EvaluationRecord) -> None:
        tracer = get_langfuse_tracer()
        score_map = {
            "bleu_score": record.metrics.bleu_score,
            "rouge1_f1": record.metrics.rouge1_f1,
            "rouge_l_f1": record.metrics.rouge_l_f1,
            "semantic_similarity": record.metrics.semantic_similarity,
            "latency_ms": record.metrics.latency_ms,
            "factuality_score": record.metrics.factuality_score,
            "grammar_correction_quality": record.metrics.grammar_correction_quality,
            "task_success_rate": record.metrics.task_success_rate,
            "tool_call_success_rate": record.metrics.tool_call_success_rate,
            "tool_call_error_rate": record.metrics.tool_call_error_rate,
            "tool_calls_count": record.metrics.tool_calls_count,
        }
        for name, value in score_map.items():
            if value is not None:
                tracer.score(trace_id=record.request_id, name=name, value=value)
        tracer.flush()

    def _git_commit(self) -> str | None:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=self.config.project_root,
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                or None
            )
        except Exception:
            return None


def _mean(values) -> float:
    clean_values = [float(value) for value in values if value is not None]
    return round(mean(clean_values), 4) if clean_values else 0.0

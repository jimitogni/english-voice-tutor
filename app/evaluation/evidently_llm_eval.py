from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import requests

from app.evaluation.models import EvaluationRecord, ToolCallRecord
from app.evaluation.service import EvaluationService
from app.observability.langfuse_client import get_langfuse_tracer


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [example.to_dict() for example in EvaluationService().load_dataset(path)]


def evaluate_dataset(
    *,
    dataset_path: Path,
    api_base_url: str,
    model_name: str | None = None,
    timeout_seconds: float = 180.0,
) -> list[EvaluationRecord]:
    service = EvaluationService()
    examples = service.load_dataset(dataset_path)
    records: list[EvaluationRecord] = []
    api_base_url = api_base_url.rstrip("/")

    for example in examples:
        request_id = f"eval-{example.id}"
        payload: dict[str, Any] = {
            "message": example.input,
            "enable_tts": False,
            "task_type": example.task_type,
            "tags": example.tags,
        }
        if model_name:
            payload["model_name"] = model_name

        started_at = perf_counter()
        answer = ""
        response_model = model_name or ""
        error: str | None = None
        retrieval_count = 0
        retrieval_error: str | None = None
        sources: list[dict[str, Any]] = []
        try:
            response = requests.post(
                f"{api_base_url}/api/chat",
                json=payload,
                headers={"X-Request-ID": request_id},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            answer = str(data.get("tutor_response", ""))
            response_model = str(data.get("model_name") or response_model)
            retrieval_count = int(data.get("retrieval_count", 0) or 0)
            raw_retrieval_error = data.get("retrieval_error")
            retrieval_error = raw_retrieval_error if isinstance(raw_retrieval_error, str) else None
            raw_sources = data.get("sources")
            if isinstance(raw_sources, list):
                sources = [source for source in raw_sources if isinstance(source, dict)]
        except Exception as exc:  # pragma: no cover - depends on live API.
            error = str(exc)

        tool_calls = []
        if retrieval_count or retrieval_error is not None:
            tool_calls.append(
                ToolCallRecord(
                    name="rag_retrieval",
                    status="error" if retrieval_error else "success",
                    error_message=retrieval_error,
                    metadata={"result_count": retrieval_count, "sources": sources},
                )
            )

        reference_context = example.reference_context
        expected_keywords = example.metadata.get("expected_context_keywords")
        if reference_context is None and isinstance(expected_keywords, list):
            reference_context = " ".join(str(keyword) for keyword in expected_keywords)

        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        record = service.evaluate_interaction(
            request_id=request_id,
            session_id=None,
            input_text=example.input,
            output_text=answer,
            expected_output=example.expected_output,
            reference_context=reference_context,
            model_name=response_model or model_name or "unknown",
            provider="ollama",
            task_type=example.task_type,
            tags=example.tags,
            latency_ms=latency_ms,
            error_message=error,
            tool_calls=tool_calls,
            task_success=error is None,
            metadata={
                "dataset_id": example.id,
                "dataset_path": str(dataset_path),
                "expected_tool_calls": example.expected_tool_calls,
                "sources": sources,
            },
        )
        records.append(record)

    return records


def save_results(results: list[EvaluationRecord], output_dir: Path) -> tuple[Path, Path]:
    service = EvaluationService()
    dataset_path = Path(results[0].metadata.get("dataset_path", "unknown")) if results else Path("unknown")
    model_name = results[0].model_name if results else None
    records_path, _summary_path = service.save_run_results(
        dataset_path=dataset_path,
        records=results,
        output_dir=output_dir,
        model_name=model_name,
    )

    csv_path = records_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        if results:
            fieldnames = list(results[0].to_dict().keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(record.to_dict() for record in results)
        else:
            writer = csv.writer(handle)
            writer.writerow(["empty"])

    return records_path, csv_path


def send_scores_to_langfuse(results: list[EvaluationRecord]) -> None:
    tracer = get_langfuse_tracer()
    for result in results:
        for name, value in result.metrics.to_dict().items():
            if value is not None:
                tracer.score(trace_id=result.request_id, name=name, value=value)
    tracer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English Voice Tutor LLM evaluation.")
    parser.add_argument(
        "--dataset",
        default="evals/english_practice_eval.jsonl",
        help="JSONL evaluation dataset path.",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost/english",
        help="Base URL for the FastAPI service.",
    )
    parser.add_argument("--model", default=None, help="Optional model name override.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output-dir", default="data/evaluation/results")
    args = parser.parse_args()

    results = evaluate_dataset(
        dataset_path=Path(args.dataset),
        api_base_url=args.api_base_url,
        model_name=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    records_path, csv_path = save_results(results, Path(args.output_dir))
    send_scores_to_langfuse(results)
    print(f"Saved JSONL results: {records_path}")
    print(f"Saved CSV results: {csv_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import requests

from app.observability.langfuse_client import get_langfuse_tracer


@dataclass(frozen=True)
class EvaluationRecord:
    request_id: str
    timestamp: str
    question: str
    expected_answer: str
    expected_context_keywords: list[str]
    category: str
    difficulty: str
    answer: str
    sources: list[str]
    latency_ms: float
    model_name: str | None
    error: str | None
    answer_is_empty: bool
    answer_too_short: bool
    answer_too_long: bool
    contains_error_message: bool
    contains_unknown_answer: bool
    response_length_chars: int
    question_length_chars: int
    expected_keyword_coverage: float
    retrieval_count: int


def load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean_line = line.strip()
            if clean_line:
                records.append(json.loads(clean_line))
    return records


def evaluate_dataset(
    *,
    dataset_path: Path,
    api_base_url: str,
    model_name: str | None = None,
    timeout_seconds: float = 180.0,
) -> list[EvaluationRecord]:
    records = load_dataset(dataset_path)
    results: list[EvaluationRecord] = []
    api_base_url = api_base_url.rstrip("/")

    for index, record in enumerate(records, start=1):
        request_id = f"eval-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{index}"
        question = str(record["question"])
        expected_answer = str(record.get("expected_answer", ""))
        expected_keywords = [str(value) for value in record.get("expected_context_keywords", [])]
        payload: dict[str, Any] = {"message": question, "enable_tts": False}
        if model_name:
            payload["model_name"] = model_name

        started_at = perf_counter()
        answer = ""
        sources: list[str] = []
        error: str | None = None
        response_model = model_name
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
            response_model = str(data.get("model_name") or response_model or "")
        except Exception as exc:  # pragma: no cover - depends on live API.
            error = str(exc)

        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        lowered_answer = answer.lower()
        results.append(
            EvaluationRecord(
                request_id=request_id,
                timestamp=datetime.now(UTC).isoformat(),
                question=question,
                expected_answer=expected_answer,
                expected_context_keywords=expected_keywords,
                category=str(record.get("category", "general")),
                difficulty=str(record.get("difficulty", "unknown")),
                answer=answer,
                sources=sources,
                latency_ms=latency_ms,
                model_name=response_model,
                error=error,
                answer_is_empty=not bool(answer.strip()),
                answer_too_short=0 < len(answer.strip()) < 20,
                answer_too_long=len(answer) > 3000,
                contains_error_message=contains_error_message(lowered_answer),
                contains_unknown_answer=contains_unknown_answer(lowered_answer),
                response_length_chars=len(answer),
                question_length_chars=len(question),
                expected_keyword_coverage=keyword_coverage(answer, expected_keywords),
                retrieval_count=0,
            )
        )

    return results


def contains_error_message(answer: str) -> bool:
    patterns = ["error", "exception", "traceback", "could not", "unavailable"]
    return any(pattern in answer for pattern in patterns)


def contains_unknown_answer(answer: str) -> bool:
    patterns = ["i don't know", "i do not know", "not sure", "cannot answer"]
    return any(pattern in answer for pattern in patterns)


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    lowered_answer = answer.lower()
    matched = sum(1 for keyword in expected_keywords if keyword.lower() in lowered_answer)
    return round(matched / len(expected_keywords), 4)


def save_results(results: list[EvaluationRecord], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    jsonl_path = output_dir / f"eval_results_{timestamp}.jsonl"
    csv_path = output_dir / f"eval_results_{timestamp}.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows(asdict(result) for result in results)

    return jsonl_path, csv_path


def send_scores_to_langfuse(results: list[EvaluationRecord]) -> None:
    tracer = get_langfuse_tracer()
    for result in results:
        tracer.score(
            trace_id=result.request_id,
            name="keyword_coverage_score",
            value=result.expected_keyword_coverage,
        )
        tracer.score(
            trace_id=result.request_id,
            name="latency_ms",
            value=result.latency_ms,
        )
        tracer.score(
            trace_id=result.request_id,
            name="answer_empty_flag",
            value=1 if result.answer_is_empty else 0,
        )
        tracer.score(
            trace_id=result.request_id,
            name="retrieval_count",
            value=result.retrieval_count,
            comment="English Voice Tutor currently does not use RAG retrieval.",
        )
    tracer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run English Voice Tutor LLM evaluation.")
    parser.add_argument(
        "--dataset",
        default="data/evaluation/datasets/sample_questions.jsonl",
        help="JSONL evaluation dataset path.",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost/english",
        help="Base URL for the web/API route, for example http://localhost/english.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default="data/evaluation/results")
    args = parser.parse_args()

    results = evaluate_dataset(
        dataset_path=Path(args.dataset),
        api_base_url=args.api_base_url,
        model_name=args.model_name,
    )
    jsonl_path, csv_path = save_results(results, Path(args.output_dir))
    send_scores_to_langfuse(results)
    print(f"Saved JSONL results: {jsonl_path}")
    print(f"Saved CSV results: {csv_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            clean_line = line.strip()
            if clean_line:
                records.append(json.loads(clean_line))
    return records


def generate_reports(results_path: Path, reports_dir: Path) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(results_path)
    report_paths = [
        reports_dir / "evidently_llm_quality_report.html",
        reports_dir / "evidently_rag_monitoring_report.html",
        reports_dir / "evidently_latency_report.html",
    ]

    generated_with_evidently = _try_generate_evidently_report(records, report_paths[0])
    if not generated_with_evidently:
        report_paths[0].write_text(_manual_quality_report(records), encoding="utf-8")

    report_paths[1].write_text(_manual_rag_report(records), encoding="utf-8")
    report_paths[2].write_text(_manual_latency_report(records), encoding="utf-8")
    return report_paths


def _try_generate_evidently_report(records: list[dict[str, Any]], path: Path) -> bool:
    if not records:
        return False
    try:
        import pandas as pd
        from evidently.metric_preset import DataQualityPreset
        from evidently.report import Report

        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=None, current_data=pd.DataFrame(records))
        report.save_html(str(path))
        return True
    except Exception:
        return False


def _manual_quality_report(records: list[dict[str, Any]]) -> str:
    total = len(records)
    empty = sum(1 for record in records if not str(record.get("output_text", "")).strip())
    error = sum(1 for record in records if record.get("error_message"))
    semantic = _safe_mean(record.get("semantic_similarity", 0.0) for record in records)
    bleu = _safe_mean(record.get("bleu_score", 0.0) for record in records)
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(record.get('task_type', '')))}</td>"
        f"<td>{html.escape(str(record.get('input_text', '')))}</td>"
        f"<td>{html.escape(str(record.get('output_text', ''))[:400])}</td>"
        f"<td>{record.get('semantic_similarity', 0)}</td>"
        f"<td>{html.escape(str(record.get('error_message') or ''))}</td>"
        "</tr>"
        for record in records
    )
    return _page(
        "English Voice Tutor LLM Quality Report",
        f"""
        <p>Total records: {total}</p>
        <p>Empty answers: {empty}</p>
        <p>Errors: {error}</p>
        <p>Mean semantic similarity: {semantic:.2f}</p>
        <p>Mean BLEU score: {bleu:.2f}</p>
        <table>
          <thead><tr><th>Task</th><th>Input</th><th>Output</th><th>Semantic Similarity</th><th>Error</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
    )


def _manual_rag_report(records: list[dict[str, Any]]) -> str:
    total = len(records)
    retrieval_counts = [_tool_result_count(record) for record in records]
    errors = sum(1 for record in records if _tool_error(record))
    success_rate = _safe_mean(record.get("tool_call_success_rate", 0.0) for record in records)
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(record.get('task_type', '')))}</td>"
        f"<td>{html.escape(str(record.get('input_text', '')))}</td>"
        f"<td>{_tool_result_count(record)}</td>"
        f"<td>{record.get('tool_call_success_rate', 0)}</td>"
        f"<td>{html.escape(str(record.get('reference_context') or ''))[:300]}</td>"
        f"<td>{html.escape(str(_tool_error(record) or ''))}</td>"
        "</tr>"
        for record in records
    )
    return _page(
        "English Voice Tutor Retrieval Report",
        f"""
        <p>Total records: {total}</p>
        <p>Mean retrieved chunks: {_safe_mean(retrieval_counts):.2f}</p>
        <p>Mean retrieval tool success rate: {success_rate:.2f}</p>
        <p>Retrieval errors: {errors}</p>
        <table>
          <thead><tr><th>Task</th><th>Input</th><th>Chunks</th><th>Tool Success Rate</th><th>Reference Context</th><th>Error</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
    )


def _manual_latency_report(records: list[dict[str, Any]]) -> str:
    latencies = [float(record.get("latency_ms", 0.0)) for record in records]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(record.get('request_id', '')))}</td>"
        f"<td>{record.get('latency_ms', 0)}</td>"
        f"<td>{html.escape(str(record.get('model_name', '')))}</td>"
        "</tr>"
        for record in records
    )
    return _page(
        "English Voice Tutor Latency Report",
        f"""
        <p>Mean latency: {_safe_mean(latencies):.2f} ms</p>
        <p>Max latency: {max(latencies) if latencies else 0:.2f} ms</p>
        <table>
          <thead><tr><th>Request ID</th><th>Latency ms</th><th>Model</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
    )


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; max-width: 1100px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f5f5f5; text-align: left; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {body}
</body>
</html>
"""


def _safe_mean(values) -> float:
    clean_values = [float(value) for value in values if value is not None]
    return mean(clean_values) if clean_values else 0.0


def _tool_result_count(record: dict[str, Any]) -> int:
    tool_calls = record.get("tool_calls")
    if not isinstance(tool_calls, list):
        return 0
    for tool_call in tool_calls:
        if isinstance(tool_call, dict) and tool_call.get("name") == "rag_retrieval":
            metadata = tool_call.get("metadata")
            if isinstance(metadata, dict):
                try:
                    return int(metadata.get("result_count", 0) or 0)
                except (TypeError, ValueError):
                    return 0
    return 0


def _tool_error(record: dict[str, Any]) -> str | None:
    tool_calls = record.get("tool_calls")
    if not isinstance(tool_calls, list):
        return None
    for tool_call in tool_calls:
        if isinstance(tool_call, dict) and tool_call.get("name") == "rag_retrieval":
            error_message = tool_call.get("error_message")
            return str(error_message) if error_message else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently-compatible HTML reports.")
    parser.add_argument("results_jsonl", help="Path to eval_results_*.jsonl")
    parser.add_argument("--reports-dir", default="data/reports/evidently")
    args = parser.parse_args()
    paths = generate_reports(Path(args.results_jsonl), Path(args.reports_dir))
    for path in paths:
        print(f"Generated report: {path}")


if __name__ == "__main__":
    main()

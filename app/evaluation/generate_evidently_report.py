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
    empty = sum(1 for record in records if record.get("answer_is_empty"))
    error = sum(1 for record in records if record.get("error"))
    coverage = _safe_mean(record.get("expected_keyword_coverage", 0.0) for record in records)
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(record.get('category', '')))}</td>"
        f"<td>{html.escape(str(record.get('question', '')))}</td>"
        f"<td>{html.escape(str(record.get('answer', ''))[:400])}</td>"
        f"<td>{record.get('expected_keyword_coverage', 0)}</td>"
        f"<td>{html.escape(str(record.get('error') or ''))}</td>"
        "</tr>"
        for record in records
    )
    return _page(
        "English Voice Tutor LLM Quality Report",
        f"""
        <p>Total records: {total}</p>
        <p>Empty answers: {empty}</p>
        <p>Errors: {error}</p>
        <p>Mean expected keyword coverage: {coverage:.2f}</p>
        <table>
          <thead><tr><th>Category</th><th>Question</th><th>Answer</th><th>Keyword Coverage</th><th>Error</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
    )


def _manual_rag_report(records: list[dict[str, Any]]) -> str:
    return _page(
        "English Voice Tutor Retrieval Report",
        """
        <p>This project is a voice tutor and does not currently use RAG retrieval.</p>
        <p>The evaluation schema keeps <code>retrieval_count</code> for future compatibility.</p>
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
    clean_values = [float(value) for value in values]
    return mean(clean_values) if clean_values else 0.0


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

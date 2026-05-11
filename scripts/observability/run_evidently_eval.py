from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.evidently_llm_eval import evaluate_dataset, save_results, send_scores_to_langfuse
from app.evaluation.generate_evidently_report import generate_reports


def main() -> None:
    dataset_path = Path("data/evaluation/datasets/sample_questions.jsonl")
    results = evaluate_dataset(
        dataset_path=dataset_path,
        api_base_url="http://localhost/english",
    )
    jsonl_path, csv_path = save_results(results, Path("data/evaluation/results"))
    send_scores_to_langfuse(results)
    print(f"Saved JSONL results: {jsonl_path}")
    print(f"Saved CSV results: {csv_path}")
    for report_path in generate_reports(jsonl_path, Path("data/reports/evidently")):
        print(f"Generated report: {report_path}")


if __name__ == "__main__":
    main()

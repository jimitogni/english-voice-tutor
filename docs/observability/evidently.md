# Evidently

Evidently evaluation is implemented as an offline batch workflow.

Run:

```bash
python scripts/observability/run_evidently_eval.py
```

This reads:

```text
data/evaluation/datasets/sample_questions.jsonl
```

It writes raw results to:

```text
data/evaluation/results/eval_results_<timestamp>.jsonl
data/evaluation/results/eval_results_<timestamp>.csv
```

It writes reports to:

```text
data/reports/evidently/evidently_llm_quality_report.html
data/reports/evidently/evidently_rag_monitoring_report.html
data/reports/evidently/evidently_latency_report.html
```

English Voice Tutor is not a RAG app today, so retrieval quality is marked as not applicable. The schema keeps `retrieval_count` for future use.

Current custom checks:

- empty answer
- too short answer
- too long answer
- contains error message
- contains unknown-answer phrase
- latency
- response length
- question length
- expected keyword coverage

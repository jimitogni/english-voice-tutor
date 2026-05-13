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

English Voice Tutor now supports optional local RAG. When `RAG_ENABLED=true`
and the Qdrant collection has been indexed, `/api/chat` returns retrieved
sources and retrieval counts. When retrieval is disabled or no collection exists,
retrieval count remains `0` and the normal tutor flow still works.

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
- retrieval count
- retrieval error
- max and mean retrieval score

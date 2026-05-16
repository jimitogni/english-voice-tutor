# English Voice Tutor Observability

This project now has app-side hooks for:

- Langfuse traces for chat requests, prompt construction, Ollama generations, and TTS.
- Prometheus metrics exposed at `/api/metrics`, including optional RAG retrieval and embedding counters.
- Evidently-compatible offline evaluation reports saved under `data/reports/evidently`, including retrieval counts and scores when RAG is enabled.
- Structured JSON logs with request IDs, session IDs, latency, model name, and status.

The homelab already has Grafana and Prometheus running in the `customer-churn-propensity-mlops` stack. Langfuse can now be enabled either by pointing the API at an existing private Langfuse deployment or by starting the bundled `docker-compose.langfuse.yml` overlay.

## Health Checks

```bash
curl http://localhost/english/api/health
curl http://localhost/english/api/observability/health
curl http://localhost/english/api/metrics
```

## Request Correlation

Every API request gets an `X-Request-ID` response header. The same request ID is used in:

- JSON logs
- Langfuse trace ID when Langfuse is enabled
- Evaluation results when the evaluation runner sends requests
- Prometheus request metrics labels through the route path and status code

## Homelab Notes

Use the public app route:

```text
http://jimitogni.duckdns.org:8888/english/
```

For local validation on the server:

```text
http://localhost/english/
```

The API container is private on `english-voice-tutor_default`. The web container is on both `english-voice-tutor_default` and `traefik_proxy`, so Prometheus can scrape metrics through:

```yaml
metrics_path: /english/api/metrics
targets:
  - english-voice-tutor-web:80
```

# Troubleshooting

## Metrics Missing In Grafana

Check the app endpoint:

```bash
curl http://localhost/english/api/metrics
```

Check Prometheus targets after adding the scrape job:

```bash
docker exec prometheus wget -qO- http://localhost:9090/api/v1/targets
```

## No Langfuse Traces

Check:

- `ENABLE_LANGFUSE_STACK=1` if you expect the local overlay to be running
- `LANGFUSE_ENABLED=true`
- `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are real project keys
- `LANGFUSE_HOST` is reachable from the API container
- `LANGFUSE_PUBLIC_URL` opens the Langfuse UI in your browser

Then run:

```bash
docker compose --env-file .env.homelab -f docker-compose.yml -f docker-compose.langfuse.yml ps
python scripts/observability/test_langfuse_trace.py
```

## Evidently Report Missing

Run:

```bash
python scripts/observability/run_evidently_eval.py
```

If Evidently import fails, the script still writes fallback HTML reports.

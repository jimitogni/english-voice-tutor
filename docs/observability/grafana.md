# Grafana And Prometheus

Grafana and Prometheus already exist on this homelab.

Detected:

- Grafana container: `grafana`
- Prometheus container: `prometheus`
- Grafana data source: `Prometheus`
- Existing Prometheus config: `/home/jimi/projects/customer-churn-propensity-mlops/monitoring/prometheus/prometheus.yml`

The safest integration is to keep the existing stack unchanged unless you choose to add a scrape target.

## Prometheus Scrape Target

Add this job to the existing Prometheus config when ready:

```yaml
  - job_name: english-voice-tutor
    metrics_path: /english/api/metrics
    static_configs:
      - targets:
          - english-voice-tutor-web:80
```

Then restart only Prometheus from its existing compose project.

## Dashboard

Import:

```text
docs/observability/grafana_llm_dashboard.json
```

The dashboard expects the existing Prometheus data source UID:

```text
prometheus
```

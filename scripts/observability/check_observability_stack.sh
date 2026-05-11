#!/usr/bin/env bash
set -euo pipefail

echo "Containers matching observability tools:"
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' \
  | awk 'NR==1 || /langfuse|grafana|prometheus|loki|tempo|jaeger|otel|opentelemetry/'

echo
echo "English Voice Tutor health:"
curl -fsS http://localhost/english/api/observability/health
echo

echo
echo "Metrics endpoint sample:"
curl -fsS http://localhost/english/api/metrics | head -40

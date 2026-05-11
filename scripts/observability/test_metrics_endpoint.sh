#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost/english}"
curl -fsS "${API_BASE_URL}/api/metrics" | grep -E 'fastapi_requests_total|llm_requests_total' || {
  echo "Expected metrics were not found." >&2
  exit 1
}

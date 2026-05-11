#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost/english}"
REQUEST_ID="${REQUEST_ID:-test-chat-$(date +%Y%m%d%H%M%S)}"

curl -fsS \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: ${REQUEST_ID}" \
  -d '{"message":"Explain the difference between although and even though in one short answer.","mode":"free","model_name":"llama3.2:3b","enable_tts":false}' \
  "${API_BASE_URL}/api/chat"
echo
echo "request_id=${REQUEST_ID}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f .env.homelab ]]; then
  echo ".env.homelab is missing. Copy .env.homelab.example first." >&2
  exit 1
fi

set -a
# shellcheck source=/dev/null
source .env.homelab
set +a

WEB_URL="http://127.0.0.1:${WEB_PORT:-8080}"
API_URL="http://127.0.0.1:${API_PORT:-8000}"
OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT:-11434}"

echo "Docker services:"
docker compose --env-file .env.homelab ps

echo
echo "Checking web: $WEB_URL"
curl -fsS "$WEB_URL" >/dev/null
echo "web ok"

echo "Checking API: $API_URL/api/status"
curl -fsS "$API_URL/api/status" >/dev/null
echo "api ok"

echo "Checking Ollama: $OLLAMA_URL/api/tags"
curl -fsS "$OLLAMA_URL/api/tags" >/dev/null
echo "ollama ok"

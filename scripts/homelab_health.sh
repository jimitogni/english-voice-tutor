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

WEB_PATH="${WEB_BASE_PATH:-/}"
if [[ -z "$WEB_PATH" || "$WEB_PATH" == "/" ]]; then
  WEB_PATH=""
else
  WEB_PATH="/${WEB_PATH#/}"
  WEB_PATH="${WEB_PATH%/}"
fi

PUBLIC_HOST="${PUBLIC_APPS_HOST:-jimitogni.duckdns.org}"
TRAEFIK_URL="http://127.0.0.1:${TRAEFIK_HOST_PORT:-80}"
TRAEFIK_HTTPS_PORT="${TRAEFIK_HTTPS_HOST_PORT:-8443}"
WEB_URL="${TRAEFIK_URL}${WEB_PATH}/"
API_URL="${TRAEFIK_URL}${WEB_PATH}/api/status"
HTTPS_API_URL="https://${PUBLIC_HOST}:${TRAEFIK_HTTPS_PORT}${WEB_PATH}/api/status"

echo "Docker services:"
docker compose --env-file .env.homelab ps

echo
echo "Checking web through Traefik: http://${PUBLIC_HOST}:8888${WEB_PATH}/"
curl -fsS -H "Host: ${PUBLIC_HOST}:8888" "$WEB_URL" >/dev/null
echo "web ok"

echo "Checking API through Traefik: http://${PUBLIC_HOST}:8888${WEB_PATH}/api/status"
curl -fsS -H "Host: ${PUBLIC_HOST}:8888" "$API_URL" >/dev/null
echo "api ok"

echo "Checking HTTPS API through Traefik: ${HTTPS_API_URL}"
curl -fsS --resolve "${PUBLIC_HOST}:${TRAEFIK_HTTPS_PORT}:127.0.0.1" "$HTTPS_API_URL" >/dev/null
echo "https api ok"

echo "Checking Ollama inside Docker"
docker compose --env-file .env.homelab exec -T ollama ollama list >/dev/null
echo "ollama ok"

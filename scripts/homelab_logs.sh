#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f .env.homelab ]]; then
  echo ".env.homelab is missing. Copy .env.homelab.example first." >&2
  exit 1
fi

docker compose --env-file .env.homelab logs -f "$@"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose is not installed. Run scripts/homelab_install_docker.sh first." >&2
  exit 1
fi

if [[ ! -f .env.homelab ]]; then
  cp .env.homelab.example .env.homelab
  echo "Created .env.homelab from .env.homelab.example."
  echo "Review it before exposing this service beyond your LAN."
fi

mkdir -p data/audio_inputs data/audio_outputs data/conversations data/knowledge data/vocabulary models/piper

if [[ "${SKIP_PIPER_DOWNLOAD:-0}" != "1" ]]; then
  bash scripts/homelab_download_piper_voices.sh
fi

compose_args=(--env-file .env.homelab -f docker-compose.yml)
if [[ "${ENABLE_GPU:-0}" == "1" || "${ENABLE_OLLAMA_GPU:-0}" == "1" ]]; then
  compose_args+=(-f docker-compose.gpu.yml)
  echo "GPU override enabled for Ollama and API services."
fi

echo "Building and starting the homelab stack..."
docker compose "${compose_args[@]}" up -d --build

set -a
# shellcheck source=/dev/null
source .env.homelab
set +a

if [[ "${SKIP_OLLAMA_PULL:-0}" != "1" ]]; then
  models_to_pull=("${OLLAMA_MODEL:-llama3.2:3b}")
  if [[ -n "${EXTRA_OLLAMA_MODELS:-}" ]]; then
    # Intentionally split on spaces for a simple .env list.
    # shellcheck disable=SC2206
    extra_models=(${EXTRA_OLLAMA_MODELS})
    models_to_pull+=("${extra_models[@]}")
  fi

  for model in "${models_to_pull[@]}"; do
    echo "Pulling Ollama model: $model"
    docker compose "${compose_args[@]}" exec -T ollama ollama pull "$model"
  done

  if [[ "${RAG_ENABLED:-false}" == "true" && -n "${RAG_EMBEDDING_MODEL:-}" ]]; then
    echo "Pulling Ollama embedding model: ${RAG_EMBEDDING_MODEL}"
    docker compose "${compose_args[@]}" exec -T ollama ollama pull "$RAG_EMBEDDING_MODEL"
  fi
fi

bash scripts/homelab_health.sh

echo
echo "Homelab deployment is ready:"
echo "  Web: http://localhost:${WEB_PORT:-8080}"
echo "  API: http://localhost:${API_PORT:-8000}/docs"

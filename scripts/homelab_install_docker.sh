#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "Docker and Docker Compose are already installed."
  docker --version
  docker compose version
  exit 0
fi

if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required to install Docker on this server." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Installing curl first..."
  sudo apt-get update
  sudo apt-get install -y curl ca-certificates
fi

echo "Installing Docker Engine and Docker Compose plugin with Docker's official installer."
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sudo sh /tmp/get-docker.sh

if getent group docker >/dev/null 2>&1; then
  sudo usermod -aG docker "$USER"
  echo "Added $USER to the docker group."
  echo "Log out and back in, or run: newgrp docker"
fi

docker --version
docker compose version

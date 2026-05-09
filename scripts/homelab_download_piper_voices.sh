#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="$PROJECT_ROOT/models/piper"

mkdir -p "$VOICE_DIR"

download_file() {
  local url="$1"
  local output="$2"

  if [[ -s "$output" ]]; then
    echo "Already present: ${output#$PROJECT_ROOT/}"
    return
  fi

  echo "Downloading ${output#$PROJECT_ROOT/}"
  curl -fL "$url" -o "$output"
}

download_voice() {
  local speaker="$1"
  local quality="$2"
  local stem="en_US-${speaker}-${quality}"
  local base_url="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/${speaker}/${quality}/${stem}"

  download_file "${base_url}.onnx" "$VOICE_DIR/${stem}.onnx"
  download_file "${base_url}.onnx.json" "$VOICE_DIR/${stem}.onnx.json"
}

download_voice "lessac" "medium"
download_voice "amy" "medium"
download_voice "ryan" "medium"

echo "Piper voices are ready in ${VOICE_DIR#$PROJECT_ROOT/}."

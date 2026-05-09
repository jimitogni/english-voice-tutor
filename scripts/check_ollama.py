from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config
from app.llm_client import OllamaError, OllamaClient


def main() -> int:
    config = load_config()
    client = OllamaClient(config)

    print(f"Checking Ollama at {config.ollama_base_url}")
    try:
        models = client.list_models()
    except OllamaError as exc:
        print(f"FAILED: {exc}")
        return 1

    print("OK: Ollama is running.")
    if models:
        print("Available models:")
        for model in models:
            marker = " <- configured" if model == config.ollama_model else ""
            print(f"  - {model}{marker}")
    else:
        print("No local models found.")

    if config.ollama_model not in models:
        print(f"\nConfigured model is missing: {config.ollama_model}")
        print(f"Run: ollama pull {config.ollama_model}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

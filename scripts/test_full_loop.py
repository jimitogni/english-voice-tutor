from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_config
from app.llm_client import OllamaError, OllamaClient
from app.memory import ConversationMemory
from app.prompts import available_modes
from app.tutor_agent import EnglishTutorAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one typed tutor interaction.")
    parser.add_argument("text", nargs="*", help="Text to send to the tutor.")
    parser.add_argument("--mode", choices=available_modes(), default="free")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    user_text = " ".join(args.text).strip() or input("Type a sentence to practice: ").strip()
    if not user_text:
        print("No input provided.")
        return 1

    config = load_config()
    client = OllamaClient(config)
    agent = EnglishTutorAgent(
        llm_client=client,
        memory=ConversationMemory(config),
        config=config,
        mode=args.mode,
    )

    try:
        response = agent.reply(user_text)
    except OllamaError as exc:
        print(f"Ollama error: {exc}")
        return 1

    print("\nTutor response:")
    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

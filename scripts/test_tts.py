from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tts import TextToSpeechEngine, TextToSpeechError


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and play one Piper TTS sample.")
    parser.add_argument(
        "text",
        nargs="*",
        default=["Hello! This is a local English tutor test."],
        help="Text to synthesize.",
    )
    args = parser.parse_args()

    try:
        output_path = TextToSpeechEngine().speak(" ".join(args.text))
    except TextToSpeechError as exc:
        print(exc)
        return 1

    print(f"Generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

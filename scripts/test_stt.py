from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.stt import SpeechToTextEngine, SpeechToTextError


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe one audio file with faster-whisper.")
    parser.add_argument("audio_path", help="Path to an audio file to transcribe.")
    args = parser.parse_args()

    try:
        text = SpeechToTextEngine().transcribe(args.audio_path)
    except SpeechToTextError as exc:
        print(exc)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

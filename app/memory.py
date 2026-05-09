from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import AppConfig, load_config
from app.utils import ensure_directory, file_timestamp, utc_timestamp


class ConversationMemoryError(RuntimeError):
    """Raised when a conversation session cannot be saved."""


@dataclass
class ConversationTurn:
    timestamp: str
    user_transcription: str
    tutor_response: str
    model_name: str
    stt_model_name: str


class ConversationMemory:
    def __init__(self, config: AppConfig | None = None, max_history_turns: int = 8) -> None:
        self.config = config or load_config()
        self.max_history_turns = max_history_turns
        self.session_id = file_timestamp()
        self.started_at = utc_timestamp()
        self.turns: list[ConversationTurn] = []

    def add_turn(
        self,
        user_text: str,
        tutor_response: str,
        model_name: str,
        stt_model_name: str = "typed-input",
    ) -> None:
        self.turns.append(
            ConversationTurn(
                timestamp=utc_timestamp(),
                user_transcription=user_text,
                tutor_response=tutor_response,
                model_name=model_name,
                stt_model_name=stt_model_name,
            )
        )

    def chat_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in self.turns[-self.max_history_turns :]:
            messages.append({"role": "user", "content": turn.user_transcription})
            messages.append({"role": "assistant", "content": turn.tutor_response})
        return messages

    def save_session(self, output_dir: Path | None = None) -> Path | None:
        if not self.turns:
            return None

        target_dir = ensure_directory(output_dir or self.config.conversations_dir)
        output_path = target_dir / f"conversation_{self.session_id}.json"
        payload = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "saved_at": utc_timestamp(),
            "turn_count": len(self.turns),
            "turns": [asdict(turn) for turn in self.turns],
        }
        temp_path = output_path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(output_path)
        except OSError as exc:
            raise ConversationMemoryError(
                f"Could not save conversation session to {output_path}."
            ) from exc

        return output_path

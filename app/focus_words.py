from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import AppConfig, load_config
from app.utils import ensure_directory, utc_timestamp


class FocusWordsError(RuntimeError):
    """Raised when focus words cannot be read or written."""


@dataclass(frozen=True)
class FocusWord:
    text: str
    created_at: str


class FocusWordsStore:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.path = self.config.focus_words_path

    def _load_entries(self) -> list[FocusWord]:
        if not self.path.exists():
            return []

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FocusWordsError(f"Could not read focus words from {self.path}.") from exc

        raw_words = payload.get("words", [])
        entries: list[FocusWord] = []
        if not isinstance(raw_words, list):
            return entries

        for item in raw_words:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            created_at = item.get("created_at")
            if isinstance(text, str) and isinstance(created_at, str):
                clean_text = self._normalize_text(text)
                if clean_text:
                    entries.append(FocusWord(text=clean_text, created_at=created_at))
        return entries

    def _save_entries(self, entries: list[FocusWord]) -> None:
        ensure_directory(self.config.vocabulary_dir)
        payload = {
            "updated_at": utc_timestamp(),
            "limit": self.config.focus_words_limit,
            "words": [asdict(entry) for entry in entries[: self.config.focus_words_limit]],
        }
        temp_path = self.path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temp_path.replace(self.path)
        except OSError as exc:
            raise FocusWordsError(f"Could not save focus words to {self.path}.") from exc

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())

    def list_entries(self) -> list[FocusWord]:
        return self._load_entries()

    def list_words(self) -> list[str]:
        return [entry.text for entry in self.list_entries()]

    def add_word(self, text: str) -> bool:
        clean_text = self._normalize_text(text)
        if not clean_text:
            raise FocusWordsError("Focus word cannot be empty.")
        if len(clean_text) > 80:
            raise FocusWordsError("Focus word is too long. Keep it under 80 characters.")

        entries = self._load_entries()
        existing = {entry.text.lower() for entry in entries}
        if clean_text.lower() in existing:
            return False
        if len(entries) >= self.config.focus_words_limit:
            raise FocusWordsError(
                f"You already have {self.config.focus_words_limit} focus words. "
                "Remove one before adding another."
            )

        entries.append(FocusWord(text=clean_text, created_at=utc_timestamp()))
        self._save_entries(entries)
        return True

    def remove_word(self, text: str) -> bool:
        target = self._normalize_text(text).lower()
        entries = self._load_entries()
        filtered = [entry for entry in entries if entry.text.lower() != target]
        if len(filtered) == len(entries):
            return False
        self._save_entries(filtered)
        return True

    def clear(self) -> None:
        self._save_entries([])

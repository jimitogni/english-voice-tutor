from __future__ import annotations

import hashlib
import re
from typing import Any

from app.config import AppConfig

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d\s().-]{7,}\d)\b")
SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd)\s*[:=]\s*['\"]?[^'\"\s,;]+"
)


def hash_identifier(value: str | None) -> str | None:
    clean_value = (value or "").strip()
    if not clean_value:
        return None
    return hashlib.sha256(clean_value.encode("utf-8")).hexdigest()[:16]


def redact_text(text: str | None, *, redact: bool) -> str | None:
    if text is None:
        return None
    if redact:
        return "[redacted]"

    clean_text = EMAIL_PATTERN.sub("[email]", text)
    clean_text = PHONE_PATTERN.sub("[phone]", clean_text)
    clean_text = SECRET_PATTERN.sub(r"\1=[redacted]", clean_text)
    return clean_text


def maybe_prompt(text: str | None, config: AppConfig) -> str | None:
    return redact_text(text, redact=config.privacy_redact_prompts)


def maybe_response(text: str | None, config: AppConfig) -> str | None:
    return redact_text(text, redact=config.privacy_redact_responses)


def sanitize_messages(messages: list[dict[str, str]], config: AppConfig) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role", "")),
            "content": maybe_prompt(str(message.get("content", "")), config) or "",
        }
        for message in messages
    ]


def summarize_value(value: Any, *, max_length: int = 500) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."

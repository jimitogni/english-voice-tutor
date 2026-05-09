from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import streamlit.components.v1 as components


class BrowserRecording(TypedDict, total=False):
    id: str
    name: str
    mime_type: str
    data_url: str
    duration_seconds: float
    auto_stopped: bool
    stop_reason: str
    error: str


_COMPONENT_DIR = Path(__file__).parent / "components" / "silence_recorder"
_silence_recorder = components.declare_component(
    "silence_recorder",
    path=str(_COMPONENT_DIR),
)


def silence_recorder(
    *,
    sample_rate: int,
    energy_threshold: float,
    silence_seconds: float,
    min_speech_seconds: float,
    max_seconds: float,
    key: str,
) -> BrowserRecording | None:
    value: Any = _silence_recorder(
        sample_rate=sample_rate,
        energy_threshold=energy_threshold,
        silence_seconds=silence_seconds,
        min_speech_seconds=min_speech_seconds,
        max_seconds=max_seconds,
        default=None,
        key=key,
    )
    if not isinstance(value, dict):
        return None
    return value

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.utils import resolve_project_path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only happens if requirements are skipped.
    load_dotenv = None


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}
STT_MODEL_SIZES = {"tiny", "base", "small", "medium"}
STT_DEVICES = {"cpu", "cuda", "auto"}
RECORD_MODES = {"fixed", "vad"}


class ConfigError(ValueError):
    """Raised when environment configuration is invalid."""


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    ollama_base_url: str
    ollama_model: str
    assistant_name: str
    user_display_name: str
    conversation_history_turns: int
    focus_words_limit: int
    stt_model_size: str
    stt_language: str
    stt_device: str
    stt_compute_type: str
    pronunciation_feedback: bool
    record_seconds: int
    record_mode: str
    sample_rate: int
    vad_energy_threshold: float
    vad_silence_seconds: float
    vad_min_speech_seconds: float
    vad_max_seconds: float
    vad_chunk_ms: int
    llm_stream: bool
    tts_engine: str
    stream_tts: bool
    piper_executable: str
    piper_model_path: Path
    piper_config_path: Path
    save_conversations: bool
    conversations_dir: Path
    audio_inputs_dir: Path
    audio_outputs_dir: Path
    vocabulary_dir: Path
    focus_words_path: Path


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False

    raise ConfigError(
        f"Invalid boolean value for {name}: {value!r}. "
        f"Use one of {sorted(TRUE_VALUES | FALSE_VALUES)}."
    )


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"Invalid integer value for {name}: {value!r}") from exc


def _get_positive_int(name: str, default: int) -> int:
    value = _get_int(name, default)
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero. Got {value!r}.")
    return value


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"Invalid float value for {name}: {value!r}") from exc


def _get_positive_float(name: str, default: float) -> float:
    value = _get_float(name, default)
    if value <= 0:
        raise ConfigError(f"{name} must be greater than zero. Got {value!r}.")
    return value


def _get_non_empty(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ConfigError(f"{name} must not be empty.")
    return value


def _get_choice(name: str, default: str, choices: set[str]) -> str:
    value = _get_non_empty(name, default).lower()
    if value not in choices:
        raise ConfigError(f"Invalid {name}: {value!r}. Use one of {sorted(choices)}.")
    return value


def load_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[1]

    if load_dotenv is not None:
        load_dotenv(project_root / ".env")

    return AppConfig(
        project_root=project_root,
        ollama_base_url=_get_non_empty("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=_get_non_empty("OLLAMA_MODEL", "llama3.2:3b"),
        assistant_name=_get_non_empty("ASSISTANT_NAME", "Jarvis"),
        user_display_name=_get_non_empty("USER_DISPLAY_NAME", "Jimi Jeday Marster"),
        conversation_history_turns=_get_positive_int("CONVERSATION_HISTORY_TURNS", 40),
        focus_words_limit=_get_positive_int("FOCUS_WORDS_LIMIT", 10),
        stt_model_size=_get_choice("STT_MODEL_SIZE", "base", STT_MODEL_SIZES),
        stt_language=_get_non_empty("STT_LANGUAGE", "en"),
        stt_device=_get_choice("STT_DEVICE", "cpu", STT_DEVICES),
        stt_compute_type=_get_non_empty("STT_COMPUTE_TYPE", "int8"),
        pronunciation_feedback=_get_bool("PRONUNCIATION_FEEDBACK", True),
        record_seconds=_get_positive_int("RECORD_SECONDS", 8),
        record_mode=_get_choice("RECORD_MODE", "fixed", RECORD_MODES),
        sample_rate=_get_positive_int("SAMPLE_RATE", 16000),
        vad_energy_threshold=_get_positive_float("VAD_ENERGY_THRESHOLD", 0.02),
        vad_silence_seconds=_get_positive_float("VAD_SILENCE_SECONDS", 1.0),
        vad_min_speech_seconds=_get_positive_float("VAD_MIN_SPEECH_SECONDS", 0.4),
        vad_max_seconds=_get_positive_float("VAD_MAX_SECONDS", 12.0),
        vad_chunk_ms=_get_positive_int("VAD_CHUNK_MS", 30),
        llm_stream=_get_bool("LLM_STREAM", True),
        tts_engine=_get_non_empty("TTS_ENGINE", "piper").lower(),
        stream_tts=_get_bool("STREAM_TTS", True),
        piper_executable=_get_non_empty("PIPER_EXECUTABLE", "piper"),
        piper_model_path=resolve_project_path(
            project_root,
            os.getenv("PIPER_MODEL_PATH", "./models/piper/en_US-lessac-medium.onnx"),
        ),
        piper_config_path=resolve_project_path(
            project_root,
            os.getenv("PIPER_CONFIG_PATH", "./models/piper/en_US-lessac-medium.onnx.json"),
        ),
        save_conversations=_get_bool("SAVE_CONVERSATIONS", True),
        conversations_dir=project_root / "data" / "conversations",
        audio_inputs_dir=project_root / "data" / "audio_inputs",
        audio_outputs_dir=project_root / "data" / "audio_outputs",
        vocabulary_dir=project_root / "data" / "vocabulary",
        focus_words_path=project_root / "data" / "vocabulary" / "focus_words.json",
    )

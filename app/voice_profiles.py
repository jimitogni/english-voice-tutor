from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from app.config import AppConfig


@dataclass(frozen=True)
class PiperVoiceProfile:
    key: str
    label: str
    model_path: Path
    config_path: Path

    @property
    def is_available(self) -> bool:
        return self.model_path.exists() and self.config_path.exists()


VOICE_FILE_STEMS: dict[str, str] = {
    "lessac": "en_US-lessac-medium",
    "amy": "en_US-amy-medium",
    "ryan": "en_US-ryan-medium",
}

VOICE_LABELS: dict[str, str] = {
    "lessac": "Lessac - US English",
    "amy": "Amy - US English",
    "ryan": "Ryan - US English",
}

MODEL_VOICE_KEYS: dict[str, str] = {
    "llama3.2:3b": "lessac",
    "qwen3:4b": "amy",
    "gemma3:4b": "ryan",
}


def voice_profile(config: AppConfig, voice_key: str) -> PiperVoiceProfile:
    stem = VOICE_FILE_STEMS.get(voice_key, VOICE_FILE_STEMS["lessac"])
    key = voice_key if voice_key in VOICE_FILE_STEMS else "lessac"
    model_path = config.project_root / "models" / "piper" / f"{stem}.onnx"
    return PiperVoiceProfile(
        key=key,
        label=VOICE_LABELS[key],
        model_path=model_path,
        config_path=model_path.with_suffix(".onnx.json"),
    )


def voice_profile_for_model(config: AppConfig, model_name: str) -> PiperVoiceProfile:
    voice_key = MODEL_VOICE_KEYS.get(model_name, "lessac")
    return voice_profile(config, voice_key)


def apply_voice_profile(config: AppConfig, profile: PiperVoiceProfile) -> AppConfig:
    return replace(
        config,
        piper_model_path=profile.model_path,
        piper_config_path=profile.config_path,
    )

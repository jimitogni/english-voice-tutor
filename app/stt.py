from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import AppConfig, load_config


class SpeechToTextError(RuntimeError):
    """Raised when speech-to-text transcription fails."""


class SpeechToTextEngine:
    """Speech-to-text backend powered by faster-whisper."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.model_size = self.config.stt_model_size
        self.language = self.config.stt_language
        self.device = self.config.stt_device
        self.compute_type = self.config.stt_compute_type
        self._model: Any | None = None

    @property
    def backend_name(self) -> str:
        return f"faster-whisper:{self.model_size}"

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise SpeechToTextError(
                "Missing speech-to-text dependency `faster-whisper`. Install it with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise SpeechToTextError(
                "Could not load the faster-whisper model "
                f"{self.model_size!r}. Check your internet connection for the first "
                "model download, disk space, and whether the model size is valid "
                "(tiny, base, small, or medium)."
            ) from exc

        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise SpeechToTextError(f"Audio file does not exist: {audio_file}")
        if not audio_file.is_file():
            raise SpeechToTextError(f"Audio path is not a file: {audio_file}")

        model = self._load_model()

        try:
            segments, _info = model.transcribe(
                str(audio_file),
                language=self.language,
                vad_filter=True,
                beam_size=5,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise SpeechToTextError(
                f"Could not transcribe {audio_file}. Make sure it is a readable WAV file."
            ) from exc

        if not text:
            raise SpeechToTextError(
                "No speech was detected in the recording. Try speaking closer to the "
                "microphone or increasing RECORD_SECONDS."
            )

        return text

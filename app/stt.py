from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import AppConfig, load_config


class SpeechToTextError(RuntimeError):
    """Raised when speech-to-text transcription fails."""


@dataclass(frozen=True)
class TranscriptionSegment:
    text: str
    start: float
    end: float
    avg_logprob: float | None
    no_speech_prob: float | None


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment]
    pronunciation_feedback: str | None = None


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

    def _build_pronunciation_feedback(self, segments: list[TranscriptionSegment]) -> str | None:
        if not self.config.pronunciation_feedback or not segments:
            return None

        low_confidence_segments = [
            segment
            for segment in segments
            if segment.avg_logprob is not None and segment.avg_logprob < -0.75
        ]
        possible_silence = [
            segment
            for segment in segments
            if segment.no_speech_prob is not None and segment.no_speech_prob > 0.6
        ]

        notes: list[str] = []
        if low_confidence_segments:
            notes.append(
                "Pronunciation note: some words were transcribed with lower confidence. "
                "Try speaking a little slower and stressing key words clearly."
            )
        if possible_silence:
            notes.append(
                "Audio note: the recording may contain silence or background noise. "
                "Try moving closer to the microphone."
            )

        if not notes:
            return "Pronunciation note: your speech was transcribed clearly."

        return " ".join(notes)

    def transcribe_detailed(self, audio_path: str | Path) -> TranscriptionResult:
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
                word_timestamps=False,
            )
            parsed_segments = [
                TranscriptionSegment(
                    text=segment.text.strip(),
                    start=float(getattr(segment, "start", 0.0)),
                    end=float(getattr(segment, "end", 0.0)),
                    avg_logprob=getattr(segment, "avg_logprob", None),
                    no_speech_prob=getattr(segment, "no_speech_prob", None),
                )
                for segment in segments
                if segment.text.strip()
            ]
            text = " ".join(segment.text for segment in parsed_segments).strip()
        except Exception as exc:
            raise SpeechToTextError(
                f"Could not transcribe {audio_file}. Make sure it is a readable audio file."
            ) from exc

        if not text:
            raise SpeechToTextError(
                "No speech was detected in the recording. Try speaking closer to the "
                "microphone or increasing RECORD_SECONDS."
            )

        return TranscriptionResult(
            text=text,
            segments=parsed_segments,
            pronunciation_feedback=self._build_pronunciation_feedback(parsed_segments),
        )

    def transcribe(self, audio_path: str | Path) -> str:
        return self.transcribe_detailed(audio_path).text

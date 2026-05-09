from __future__ import annotations

import wave
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import AppConfig, load_config
from app.utils import ensure_directory, file_timestamp


class AudioRecordingError(RuntimeError):
    """Raised when microphone recording is unavailable or fails."""


class AudioRecorder:
    """Record mono WAV audio from the default microphone."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.sample_rate = self.config.sample_rate
        self.default_duration_seconds = self.config.record_seconds
        self.output_dir = ensure_directory(self.config.audio_inputs_dir)

    def _load_dependencies(self) -> tuple[Any, Any]:
        try:
            import numpy as np
        except ImportError as exc:
            raise AudioRecordingError(
                "Missing Python dependency `numpy`. Install project dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioRecordingError(
                "Missing Python dependency `sounddevice`. Install project dependencies "
                "with `python -m pip install -r requirements.txt`."
            ) from exc
        except OSError as exc:
            raise AudioRecordingError(
                "The `sounddevice` package is installed, but PortAudio is missing. "
                "On Debian/Ubuntu, install it with "
                "`sudo apt install libportaudio2 portaudio19-dev`."
            ) from exc
        return np, sd

    def _ensure_microphone_available(self, sd: Any) -> None:
        try:
            device = sd.query_devices(kind="input")
        except Exception as exc:
            raise AudioRecordingError(
                "Could not find a default input microphone. Check your system audio "
                "settings, connect a microphone, or test devices with `python -m "
                "sounddevice`."
            ) from exc

        if not device or int(device.get("max_input_channels", 0)) < 1:
            raise AudioRecordingError(
                "The default audio input device does not expose any input channels. "
                "Select another microphone in your system audio settings."
            )

    def _build_output_path(self) -> Path:
        suffix = uuid4().hex[:8]
        filename = f"input_{file_timestamp()}_{suffix}.wav"
        return self.output_dir / filename

    def record(self, duration_seconds: float | None = None) -> Path:
        duration = float(duration_seconds or self.default_duration_seconds)
        if duration <= 0:
            raise AudioRecordingError("Recording duration must be greater than zero seconds.")

        np, sd = self._load_dependencies()
        self._ensure_microphone_available(sd)

        frames = max(1, int(duration * self.sample_rate))
        output_path = self._build_output_path()

        try:
            audio = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except Exception as exc:
            raise AudioRecordingError(
                "Microphone recording failed. Check that no other app is blocking the "
                "microphone and that PortAudio can access your input device."
            ) from exc

        pcm_audio = np.clip(audio.reshape(-1), -1.0, 1.0)
        pcm_audio = (pcm_audio * 32767).astype(np.int16)

        try:
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_audio.tobytes())
        except OSError as exc:
            raise AudioRecordingError(f"Could not save recorded audio to {output_path}.") from exc

        return output_path

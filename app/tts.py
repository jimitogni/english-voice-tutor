from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import wave
from collections.abc import Iterable, Iterator
from pathlib import Path
from uuid import uuid4

from app.config import AppConfig, load_config
from app.utils import ensure_directory, file_timestamp


class TextToSpeechError(RuntimeError):
    """Raised when text-to-speech synthesis fails."""


def speech_text_from_markdown(text: str) -> str:
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    clean_text = re.sub(r"^\s{0,3}#{1,6}\s+", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^\s{0,3}[-*+]\s+", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^\s{0,3}\d+[.)]\s+", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^\s{0,3}>\s?", "", clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", clean_text)
    clean_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean_text)
    clean_text = re.sub(r"`([^`]+)`", r"\1", clean_text)
    clean_text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", clean_text)
    clean_text = re.sub(r"__([^_\n]+)__", r"\1", clean_text)
    clean_text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", clean_text)
    clean_text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", r"\1", clean_text)
    clean_text = clean_text.replace("*", "")
    clean_text = re.sub(r"[ \t]+", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text.strip()


class TextToSpeechEngine:
    """Text-to-speech backend powered by Piper."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or load_config()
        self.engine = self.config.tts_engine.strip().lower()
        self.output_dir = ensure_directory(self.config.audio_outputs_dir)

    def _resolve_piper_executable(self) -> str:
        configured = self.config.piper_executable.strip()
        if not configured:
            raise TextToSpeechError(
                "PIPER_EXECUTABLE is empty. Set it to `piper` or an absolute path "
                "to the Piper executable."
            )

        configured_path = Path(configured).expanduser()
        looks_like_path = configured_path.is_absolute() or os.sep in configured
        if looks_like_path:
            candidates = [configured_path]
            if not configured_path.is_absolute():
                candidates.insert(0, self.config.project_root / configured_path)

            for candidate in candidates:
                if candidate.exists() and os.access(candidate, os.X_OK):
                    return str(candidate)

            raise TextToSpeechError(
                "Piper executable was not found or is not executable at "
                f"{configured!r}. Install Piper or update PIPER_EXECUTABLE in `.env`."
            )

        executable = shutil.which(configured)
        if executable is not None:
            return executable

        interpreter_candidates = [
            Path(sys.executable).parent / configured,
            Path(sys.prefix) / "bin" / configured,
        ]
        for candidate in interpreter_candidates:
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)

        raise TextToSpeechError(
            "Piper executable was not found on PATH or beside the active Python "
            "interpreter. Install it with `python -m pip install piper-tts` inside "
            "the virtual environment, activate the venv, or set PIPER_EXECUTABLE "
            "in `.env`."
        )

    def _check_voice_files(self) -> None:
        if not self.config.piper_model_path.exists():
            raise TextToSpeechError(
                "Piper voice model file is missing: "
                f"{self.config.piper_model_path}. Download a `.onnx` voice model "
                "or update PIPER_MODEL_PATH in `.env`."
            )

        if not self.config.piper_config_path.exists():
            raise TextToSpeechError(
                "Piper voice config file is missing: "
                f"{self.config.piper_config_path}. Download the matching "
                "`.onnx.json` file or update PIPER_CONFIG_PATH in `.env`."
            )

    def _build_output_path(self) -> Path:
        return self.output_dir / f"tts_{file_timestamp()}_{uuid4().hex[:8]}.wav"

    def _run_piper_command(self, command: list[str], clean_text: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                command,
                input=clean_text,
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TextToSpeechError(
                "Piper timed out while generating speech. Try a shorter response, "
                "a smaller voice model, or check CPU load."
            ) from exc

    def synthesize(self, text: str) -> Path:
        if self.engine in {"none", "off", "disabled"}:
            raise TextToSpeechError("TTS is disabled because TTS_ENGINE is set to none/off.")

        if self.engine != "piper":
            raise TextToSpeechError(
                f"Unsupported TTS_ENGINE {self.config.tts_engine!r}. "
                "The Phase 3 backend currently supports `piper`."
            )

        clean_text = speech_text_from_markdown(text)
        if not clean_text:
            raise TextToSpeechError("Cannot synthesize empty text.")

        self._check_voice_files()
        executable = self._resolve_piper_executable()
        output_path = self._build_output_path()

        command = [
            executable,
            "--model",
            str(self.config.piper_model_path),
            "--config",
            str(self.config.piper_config_path),
            "--output_file",
            str(output_path),
        ]
        if self.config.piper_cuda:
            command.insert(1, "--cuda")

        result = self._run_piper_command(command, clean_text)

        if result.returncode != 0 and "--config" in result.stderr:
            fallback_command = [
                executable,
                "--model",
                str(self.config.piper_model_path),
                "--output_file",
                str(output_path),
            ]
            if self.config.piper_cuda:
                fallback_command.insert(1, "--cuda")
            result = self._run_piper_command(fallback_command, clean_text)

        if result.returncode != 0:
            details = (result.stderr or result.stdout or "no Piper output").strip()
            raise TextToSpeechError(f"Piper failed to generate speech: {details[:800]}")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise TextToSpeechError(
                f"Piper finished but did not create a valid WAV file at {output_path}."
            )

        return output_path

    def play(self, audio_path: str | Path) -> None:
        wav_path = Path(audio_path)
        if not wav_path.exists():
            raise TextToSpeechError(f"Cannot play missing audio file: {wav_path}")

        player_commands = [
            ("aplay", ["aplay", "-q", str(wav_path)]),
            ("paplay", ["paplay", str(wav_path)]),
            ("pw-play", ["pw-play", str(wav_path)]),
            ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(wav_path)]),
        ]

        for executable, command in player_commands:
            if shutil.which(executable) is None:
                continue

            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                continue

            if result.returncode == 0:
                return

        self._play_with_sounddevice(wav_path)

    def _play_with_sounddevice(self, wav_path: Path) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise TextToSpeechError(
                "No audio playback command was found (`aplay`, `paplay`, `pw-play`, "
                "or `ffplay`) and Python playback dependencies are missing. "
                "Install `alsa-utils` or reinstall project dependencies."
            ) from exc
        except OSError as exc:
            raise TextToSpeechError(
                "No audio playback command was found, and `sounddevice` cannot load "
                "PortAudio. Install playback tools with `sudo apt install alsa-utils` "
                "or PortAudio with `sudo apt install libportaudio2 portaudio19-dev`."
            ) from exc

        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_rate = wav_file.getframerate()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
        except wave.Error as exc:
            raise TextToSpeechError(f"Could not read WAV file for playback: {wav_path}") from exc

        if sample_width != 2:
            raise TextToSpeechError(
                f"Unsupported WAV sample width {sample_width}. Expected 16-bit PCM."
            )

        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)

        try:
            sd.play(audio, samplerate=sample_rate)
            sd.wait()
        except Exception as exc:
            raise TextToSpeechError("Audio playback failed through sounddevice.") from exc

    def speak(self, text: str) -> Path:
        output_path = self.synthesize(text)
        self.play(output_path)
        return output_path

    def iter_speech_chunks(self, text_chunks: Iterable[str]) -> Iterator[str]:
        buffer = ""
        sentence_pattern = re.compile(r"^(.+?[.!?])(\s+|$)", re.DOTALL)

        for text_chunk in text_chunks:
            buffer += text_chunk

            while True:
                match = sentence_pattern.match(buffer)
                if match is None:
                    break

                sentence = match.group(1).strip()
                buffer = buffer[match.end() :].lstrip()
                if sentence:
                    yield sentence

        remainder = buffer.strip()
        if remainder:
            yield remainder

    def speak_stream(self, text_chunks: Iterable[str]) -> Iterator[Path]:
        for speech_chunk in self.iter_speech_chunks(text_chunks):
            yield self.speak(speech_chunk)

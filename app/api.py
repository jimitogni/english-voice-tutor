from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import cast
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import AppConfig, ConfigError, load_config
from app.focus_words import FocusWordsError, FocusWordsStore
from app.llm_client import OllamaClient, OllamaError
from app.memory import ConversationMemory, ConversationMemoryError
from app.prompts import TutorMode, available_modes, get_mode_definition
from app.stt import SpeechToTextEngine, SpeechToTextError, TranscriptionResult
from app.tts import TextToSpeechEngine, TextToSpeechError
from app.tutor_agent import EnglishTutorAgent
from app.utils import ensure_directory, file_timestamp
from app.voice_profiles import (
    VOICE_FILE_STEMS,
    PiperVoiceProfile,
    apply_voice_profile,
    voice_profile,
    voice_profile_for_model,
)


RECOMMENDED_OLLAMA_MODELS = ["llama3.2:3b", "qwen3:4b", "gemma3:4b"]
SUPPORTED_UPLOAD_SUFFIXES = {".wav", ".webm", ".ogg", ".m4a", ".mp3", ".mp4"}


class ModeInfo(BaseModel):
    key: str
    label: str
    description: str


class VoiceInfo(BaseModel):
    key: str
    label: str
    model_path: str
    config_path: str
    available: bool


class VadSettings(BaseModel):
    energy_threshold: float
    silence_seconds: float
    min_speech_seconds: float
    max_seconds: float
    chunk_ms: int
    sample_rate: int


class StatusResponse(BaseModel):
    assistant_name: str
    user_display_name: str
    default_model: str
    recommended_models: list[str]
    installed_models: list[str]
    ollama_error: str | None = None
    modes: list[ModeInfo]
    model_voices: dict[str, VoiceInfo]
    voice_profiles: list[VoiceInfo]
    default_mode: str
    context_turns: int
    focus_words: list[str]
    focus_words_limit: int
    tts_enabled: bool
    llm_stream_enabled: bool
    vad: VadSettings


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    mode: str = "free"
    model_name: str | None = None
    enable_tts: bool = True


class ChatResponse(BaseModel):
    session_id: str
    mode: str
    model_name: str
    voice: VoiceInfo | None
    user_text: str
    tutor_response: str
    audio_url: str | None = None
    tts_error: str | None = None
    pronunciation_feedback: str | None = None


class FocusWordsResponse(BaseModel):
    words: list[str]
    limit: int


class FocusWordRequest(BaseModel):
    text: str = Field(min_length=1, max_length=80)


class ResetRequest(BaseModel):
    session_id: str | None = None


class ResetResponse(BaseModel):
    session_id: str
    reset: bool


@dataclass
class SessionState:
    memory: ConversationMemory


sessions: dict[str, SessionState] = {}
sessions_lock = Lock()


app = FastAPI(
    title="English Voice Tutor API",
    description="Local FastAPI backend for the English voice tutor.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_config_or_500() -> AppConfig:
    try:
        return load_config()
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=f"Configuration error: {exc}") from exc


def _voice_info(profile: PiperVoiceProfile) -> VoiceInfo:
    return VoiceInfo(
        key=profile.key,
        label=profile.label,
        model_path=str(profile.model_path),
        config_path=str(profile.config_path),
        available=profile.is_available,
    )


def _configured_voice_config(config: AppConfig, model_name: str) -> tuple[AppConfig, PiperVoiceProfile]:
    model_config = replace(config, ollama_model=model_name)
    profile = voice_profile_for_model(model_config, model_name)
    return apply_voice_profile(model_config, profile), profile


def _parse_mode(mode: str) -> TutorMode:
    normalized = mode.strip().lower()
    if normalized not in available_modes():
        allowed = ", ".join(available_modes())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tutor mode {mode!r}. Available modes: {allowed}.",
        )
    return cast(TutorMode, normalized)


def _session_id(session_id: str | None) -> str:
    clean_session_id = (session_id or "").strip()
    if clean_session_id:
        return clean_session_id
    return uuid4().hex


def _get_session(session_id: str, config: AppConfig) -> SessionState:
    with sessions_lock:
        state = sessions.get(session_id)
        if state is None:
            state = SessionState(memory=ConversationMemory(config))
            sessions[session_id] = state
        return state


def _save_session_if_enabled(state: SessionState, config: AppConfig) -> None:
    if not config.save_conversations:
        return
    try:
        state.memory.save_session(config.conversations_dir)
    except ConversationMemoryError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _agent_for_request(
    *,
    config: AppConfig,
    session_id: str,
    mode: TutorMode,
) -> tuple[EnglishTutorAgent, SessionState]:
    state = _get_session(session_id, config)
    agent = EnglishTutorAgent(
        llm_client=OllamaClient(config),
        memory=state.memory,
        config=config,
        mode=mode,
    )
    return agent, state


def _synthesize_response(config: AppConfig, text: str, enable_tts: bool) -> tuple[str | None, str | None]:
    if not enable_tts or config.tts_engine in {"none", "off", "disabled"}:
        return None, None

    try:
        audio_path = TextToSpeechEngine(config).synthesize(text)
    except TextToSpeechError as exc:
        return None, str(exc)

    return f"/api/audio/{audio_path.name}", None


def _chat_response(
    *,
    config: AppConfig,
    session_id: str,
    mode: TutorMode,
    user_text: str,
    stt_model_name: str,
    enable_tts: bool,
    pronunciation_feedback: str | None = None,
) -> ChatResponse:
    agent, state = _agent_for_request(config=config, session_id=session_id, mode=mode)

    try:
        tutor_response = agent.reply(user_text, stt_model_name=stt_model_name)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=f"Ollama error: {exc}") from exc

    _save_session_if_enabled(state, config)
    voice = voice_profile_for_model(config, config.ollama_model)
    audio_url, tts_error = _synthesize_response(config, tutor_response, enable_tts)

    return ChatResponse(
        session_id=session_id,
        mode=mode,
        model_name=config.ollama_model,
        voice=_voice_info(voice) if enable_tts else None,
        user_text=user_text,
        tutor_response=tutor_response,
        audio_url=audio_url,
        tts_error=tts_error,
        pronunciation_feedback=pronunciation_feedback,
    )


def _model_choices(config: AppConfig, installed_models: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for model in [config.ollama_model, *RECOMMENDED_OLLAMA_MODELS, *installed_models]:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def _detect_suffix(upload: UploadFile) -> str:
    filename_suffix = Path(upload.filename or "").suffix.lower()
    if filename_suffix in SUPPORTED_UPLOAD_SUFFIXES:
        return filename_suffix

    content_type = upload.content_type or ""
    guessed_suffix = mimetypes.guess_extension(content_type.split(";")[0].strip())
    if guessed_suffix and guessed_suffix.lower() in SUPPORTED_UPLOAD_SUFFIXES:
        return guessed_suffix.lower()

    return ".webm"


def _save_upload(config: AppConfig, upload: UploadFile) -> Path:
    try:
        audio_bytes = upload.file.read()
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Could not read uploaded audio file.") from exc

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")

    input_dir = ensure_directory(config.audio_inputs_dir)
    digest = hashlib.sha256(audio_bytes).hexdigest()[:8]
    suffix = _detect_suffix(upload)
    audio_path = input_dir / f"browser_{file_timestamp()}_{digest}{suffix}"

    try:
        audio_path.write_bytes(audio_bytes)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save audio to {audio_path}.") from exc

    return audio_path


@app.get("/api/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    config = _load_config_or_500()
    installed_models: list[str] = []
    ollama_error: str | None = None
    try:
        installed_models = OllamaClient(config).list_models()
    except OllamaError as exc:
        ollama_error = str(exc)

    model_names = _model_choices(config, installed_models)
    model_voices = {
        model: _voice_info(voice_profile_for_model(config, model)) for model in model_names
    }
    all_voices = [_voice_info(voice_profile(config, key)) for key in VOICE_FILE_STEMS]

    try:
        focus_words = FocusWordsStore(config).list_words()
    except FocusWordsError:
        focus_words = []

    return StatusResponse(
        assistant_name=config.assistant_name,
        user_display_name=config.user_display_name,
        default_model=config.ollama_model,
        recommended_models=RECOMMENDED_OLLAMA_MODELS,
        installed_models=installed_models,
        ollama_error=ollama_error,
        modes=[
            ModeInfo(
                key=mode,
                label=get_mode_definition(mode).label,
                description=get_mode_definition(mode).description,
            )
            for mode in available_modes()
        ],
        model_voices=model_voices,
        voice_profiles=all_voices,
        default_mode="free",
        context_turns=config.conversation_history_turns,
        focus_words=focus_words,
        focus_words_limit=config.focus_words_limit,
        tts_enabled=config.tts_engine == "piper",
        llm_stream_enabled=config.llm_stream,
        vad=VadSettings(
            energy_threshold=config.vad_energy_threshold,
            silence_seconds=config.vad_silence_seconds,
            min_speech_seconds=config.vad_min_speech_seconds,
            max_seconds=config.vad_max_seconds,
            chunk_ms=config.vad_chunk_ms,
            sample_rate=config.sample_rate,
        ),
    )


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(request: ChatRequest) -> ChatResponse:
    base_config = _load_config_or_500()
    model_name = (request.model_name or base_config.ollama_model).strip()
    config, _profile = _configured_voice_config(base_config, model_name)
    mode = _parse_mode(request.mode)
    session_id = _session_id(request.session_id)
    user_text = request.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    return _chat_response(
        config=config,
        session_id=session_id,
        mode=mode,
        user_text=user_text,
        stt_model_name="typed-input",
        enable_tts=request.enable_tts,
    )


@app.post("/api/voice", response_model=ChatResponse)
def post_voice(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None),
    mode: str = Form(default="free"),
    model_name: str | None = Form(default=None),
    enable_tts: bool = Form(default=True),
) -> ChatResponse:
    base_config = _load_config_or_500()
    selected_model = (model_name or base_config.ollama_model).strip()
    config, _profile = _configured_voice_config(base_config, selected_model)
    parsed_mode = _parse_mode(mode)
    selected_session_id = _session_id(session_id)
    audio_path = _save_upload(config, file)

    stt_engine = SpeechToTextEngine(config)
    try:
        transcription: TranscriptionResult = stt_engine.transcribe_detailed(audio_path)
    except SpeechToTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _chat_response(
        config=config,
        session_id=selected_session_id,
        mode=parsed_mode,
        user_text=transcription.text,
        stt_model_name=stt_engine.backend_name,
        enable_tts=enable_tts,
        pronunciation_feedback=transcription.pronunciation_feedback,
    )


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    config = _load_config_or_500()
    clean_name = Path(filename).name
    if clean_name != filename:
        raise HTTPException(status_code=400, detail="Invalid audio filename.")

    audio_path = config.audio_outputs_dir / clean_name
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(audio_path, media_type="audio/wav", filename=clean_name)


@app.get("/api/focus-words", response_model=FocusWordsResponse)
def get_focus_words() -> FocusWordsResponse:
    config = _load_config_or_500()
    try:
        words = FocusWordsStore(config).list_words()
    except FocusWordsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FocusWordsResponse(words=words, limit=config.focus_words_limit)


@app.post("/api/focus-words", response_model=FocusWordsResponse)
def add_focus_word(request: FocusWordRequest) -> FocusWordsResponse:
    config = _load_config_or_500()
    store = FocusWordsStore(config)
    try:
        store.add_word(request.text)
        words = store.list_words()
    except FocusWordsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FocusWordsResponse(words=words, limit=config.focus_words_limit)


@app.delete("/api/focus-words/{word}", response_model=FocusWordsResponse)
def remove_focus_word(word: str) -> FocusWordsResponse:
    config = _load_config_or_500()
    store = FocusWordsStore(config)
    try:
        store.remove_word(word)
        words = store.list_words()
    except FocusWordsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FocusWordsResponse(words=words, limit=config.focus_words_limit)


@app.post("/api/reset", response_model=ResetResponse)
def reset_session(request: ResetRequest) -> ResetResponse:
    session_id = _session_id(request.session_id)
    with sessions_lock:
        existed = sessions.pop(session_id, None) is not None
    return ResetResponse(session_id=session_id, reset=existed)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "English Voice Tutor API",
        "docs": "/docs",
        "status": "/api/status",
        "frontend": "Run the React app from the web/ directory with npm run dev.",
    }


frontend_dist = Path(__file__).resolve().parents[1] / "web" / "dist"
if frontend_dist.exists():
    app.mount("/app", StaticFiles(directory=frontend_dist, html=True), name="react-app")


__all__ = ["app"]

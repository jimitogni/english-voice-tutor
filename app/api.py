from __future__ import annotations

import json
import hashlib
import logging
import mimetypes
from time import perf_counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import cast
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import AppConfig, ConfigError, load_config
from app.evaluation.service import EvaluationService
from app.evaluation.models import ToolCallRecord
from app.focus_words import FocusWordsError, FocusWordsStore
from app.llm_client import OllamaClient, OllamaError
from app.memory import ConversationMemory, ConversationMemoryError
from app.observability import ObservabilityContext, current_endpoint, current_request_id, current_trace
from app.observability.context import use_observability_context
from app.observability.langfuse_client import get_langfuse_tracer, langfuse_credentials_configured
from app.observability.logging import configure_logging, log_event
from app.observability.metrics import (
    metrics_response,
    observe_evaluation_interaction,
    observe_evaluation_tool_call,
    observe_fastapi_request,
    observe_stt_call,
    observe_tts_call,
    observe_user_feedback,
)
from app.observability.privacy import hash_identifier, maybe_response
from app.prompts import TutorMode, available_modes, get_mode_definition
from app.rag import RagSource
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

logger = logging.getLogger(__name__)


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


class RagStatus(BaseModel):
    enabled: bool
    vector_db: str
    embedding_model: str
    collection: str
    top_k: int
    score_threshold: float
    qdrant_url: str
    knowledge_dir: str


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
    stt_device: str
    stt_compute_type: str
    piper_cuda: bool
    vad: VadSettings
    rag: RagStatus


class ApiHealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    langfuse_enabled: bool
    evidently_enabled: bool
    evaluation_enabled: bool
    metrics_enabled: bool
    prometheus_enabled: bool
    grafana_enabled: bool


class ObservabilityHealthResponse(ApiHealthResponse):
    langfuse_configured: bool
    ollama_status: str
    stt_device: str
    stt_compute_type: str
    piper_cuda: bool


class ObservabilityRunSummary(BaseModel):
    run_id: str
    created_at: str
    dataset_path: str
    dataset_size: int
    model_name: str | None = None
    git_commit: str | None = None
    records_path: str
    summary_path: str
    averages: dict[str, float]
    counts: dict[str, int]


class ObservabilitySummaryResponse(BaseModel):
    status: str
    service: str
    environment: str
    langfuse_enabled: bool
    langfuse_url: str | None = None
    evaluation_enabled: bool
    metrics_enabled: bool
    prometheus_enabled: bool
    rag_enabled: bool
    total_interactions: int
    interactions_last_24h: int
    total_errors: int
    average_latency_ms: float
    average_feedback_score: float | None = None
    tool_call_count: int
    tool_call_error_count: int
    task_success_rate: float | None = None
    last_interaction_at: str | None = None
    latest_run: ObservabilityRunSummary | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    mode: str = "free"
    model_name: str | None = None
    enable_tts: bool = True
    expected_output: str | None = None
    reference_context: str | None = None
    task_type: str = "general"
    tags: list[str] = Field(default_factory=list)
    task_success: bool | None = None


class RagSourceInfo(BaseModel):
    title: str
    source: str
    score: float | None = None
    content_preview: str


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    mode: str
    model_name: str
    voice: VoiceInfo | None
    user_text: str
    tutor_response: str
    audio_url: str | None = None
    tts_error: str | None = None
    pronunciation_feedback: str | None = None
    sources: list[RagSourceInfo] = Field(default_factory=list)
    retrieval_count: int = 0
    retrieval_error: str | None = None


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


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=5.0)
    comment: str | None = Field(default=None, max_length=1000)


class FeedbackResponse(BaseModel):
    request_id: str
    saved: bool


@dataclass
class SessionState:
    memory: ConversationMemory


@dataclass
class CachedSttEngine:
    engine: SpeechToTextEngine
    lock: Lock


sessions: dict[str, SessionState] = {}
sessions_lock = Lock()
stt_engines: dict[tuple[str, str, str, str], CachedSttEngine] = {}
stt_engines_lock = Lock()


app = FastAPI(
    title="English Voice Tutor API",
    description="Local FastAPI backend for the English voice tutor.",
    version="0.1.0",
)

try:
    configure_logging(load_config())
except ConfigError:
    logging.basicConfig(level=logging.INFO)

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


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    config = _load_config_or_500()
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    endpoint = request.url.path
    started_at = perf_counter()
    status_code = 500

    with use_observability_context(
        ObservabilityContext(request_id=request_id, endpoint=endpoint)
    ):
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            latency_seconds = perf_counter() - started_at
            if config.metrics_enabled and config.prometheus_enabled:
                observe_fastapi_request(
                    method=request.method,
                    path=endpoint,
                    status_code=status_code,
                    latency_seconds=latency_seconds,
                )
            log_event(
                logger,
                "api_request",
                config=config,
                level=logging.ERROR,
                method=request.method,
                path=endpoint,
                status_code=status_code,
                latency_ms=round(latency_seconds * 1000, 2),
                status="error",
            )
            raise

        latency_seconds = perf_counter() - started_at
        if config.metrics_enabled and config.prometheus_enabled:
            observe_fastapi_request(
                method=request.method,
                path=endpoint,
                status_code=status_code,
                latency_seconds=latency_seconds,
            )
        log_event(
            logger,
            "api_request",
            config=config,
            method=request.method,
            path=endpoint,
            status_code=status_code,
            latency_ms=round(latency_seconds * 1000, 2),
            status="success" if status_code < 500 else "error",
        )
        response.headers["X-Request-ID"] = request_id
        return response


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


def _rag_source_info(source: RagSource) -> RagSourceInfo:
    preview = " ".join(source.content.split())
    if len(preview) > 280:
        preview = f"{preview[:277].rstrip()}..."
    return RagSourceInfo(
        title=source.title,
        source=source.source,
        score=source.score,
        content_preview=preview,
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

    started_at = perf_counter()
    trace = get_langfuse_tracer()
    try:
        audio_path = TextToSpeechEngine(config).synthesize(text)
    except TextToSpeechError as exc:
        latency_seconds = perf_counter() - started_at
        if config.metrics_enabled and config.prometheus_enabled:
            observe_tts_call(engine=config.tts_engine, status="error", latency_seconds=latency_seconds)
        trace.log_span(
            current_trace(),
            name="tts",
            metadata={"engine": config.tts_engine, "status": "error", "error": str(exc)},
            started_at=started_at,
        )
        log_event(
            logger,
            "tts_call",
            config=config,
            engine=config.tts_engine,
            status="error",
            latency_ms=round(latency_seconds * 1000, 2),
            error_message=str(exc),
        )
        return None, str(exc)

    latency_seconds = perf_counter() - started_at
    if config.metrics_enabled and config.prometheus_enabled:
        observe_tts_call(engine=config.tts_engine, status="success", latency_seconds=latency_seconds)
    trace.log_span(
        current_trace(),
        name="tts",
        input_value=maybe_response(text, config),
        output_value=str(audio_path.name),
        metadata={"engine": config.tts_engine, "piper_cuda": config.piper_cuda, "status": "success"},
        started_at=started_at,
    )
    log_event(
        logger,
        "tts_call",
        config=config,
        engine=config.tts_engine,
        piper_cuda=config.piper_cuda,
        status="success",
        latency_ms=round(latency_seconds * 1000, 2),
        audio_file=audio_path.name,
    )
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
    expected_output: str | None = None,
    reference_context: str | None = None,
    task_type: str = "general",
    tags: list[str] | None = None,
    task_success: bool | None = None,
) -> ChatResponse:
    agent, state = _agent_for_request(config=config, session_id=session_id, mode=mode)
    request_id = current_request_id() or uuid4().hex
    user_id_hash = hash_identifier(config.user_display_name) if config.privacy_hash_user_id else None
    tracer = get_langfuse_tracer()
    trace = tracer.create_trace(
        name="voice_tutor_chat",
        request_id=request_id,
        session_id=session_id,
        user_id_hash=user_id_hash,
        input_text=user_text,
        metadata={
            "service": config.service_name,
            "environment": config.observability_env,
            "endpoint": current_endpoint(),
            "mode": mode,
            "model": config.ollama_model,
            "stt_model_name": stt_model_name,
            "tts_enabled": enable_tts,
        },
    )

    with use_observability_context(
        ObservabilityContext(
            request_id=request_id,
            endpoint=current_endpoint() or "chat",
            session_id=session_id,
            user_id_hash=user_id_hash,
            trace=trace,
        )
    ):
        started_at = perf_counter()
        try:
            tutor_response = agent.reply(user_text, stt_model_name=stt_model_name)
        except OllamaError as exc:
            latency_ms = round((perf_counter() - started_at) * 1000, 2)
            llm_metadata = agent.llm_client.last_call_metadata
            evaluation_service = EvaluationService(config)
            error_record = evaluation_service.evaluate_interaction(
                request_id=request_id,
                session_id=session_id,
                input_text=user_text,
                output_text="",
                expected_output=expected_output,
                reference_context=reference_context,
                model_name=config.ollama_model,
                provider=llm_metadata.provider if llm_metadata is not None else "ollama",
                task_type=task_type,
                tags=tags,
                latency_ms=latency_ms,
                input_tokens=llm_metadata.input_tokens if llm_metadata is not None else None,
                output_tokens=llm_metadata.output_tokens if llm_metadata is not None else None,
                token_source=llm_metadata.token_source if llm_metadata is not None else None,
                error_message=str(exc),
                task_success=False if task_success is None else task_success,
                metadata={
                    "mode": mode,
                    "stt_model_name": stt_model_name,
                    "request_source": "api",
                },
            )
            evaluation_service.persist_interaction(error_record)
            if config.metrics_enabled and config.prometheus_enabled:
                observe_evaluation_interaction(
                    task_type=task_type,
                    status=error_record.workflow_status,
                    success=error_record.metrics.task_success_rate == 1.0,
                )
            tracer.update_trace(
                trace,
                metadata={"status": "error", "error_message": str(exc)},
            )
            raise HTTPException(status_code=503, detail=f"Ollama error: {exc}") from exc

        _save_session_if_enabled(state, config)
        retrieval = agent.last_retrieval
        voice = voice_profile_for_model(config, config.ollama_model)
        audio_url, tts_error = _synthesize_response(config, tutor_response, enable_tts)
        latency_seconds = perf_counter() - started_at
        llm_metadata = agent.llm_client.last_call_metadata
        tool_calls: list[ToolCallRecord] = []
        if agent.rag_retriever is not None:
            tool_status = "error" if retrieval.error else "success"
            tool_calls.append(
                ToolCallRecord(
                    name="rag_retrieval",
                    status=tool_status,
                    latency_ms=round(retrieval.latency_seconds * 1000, 2),
                    error_message=retrieval.error,
                    metadata={
                        "vector_db": retrieval.vector_db,
                        "result_count": retrieval.count,
                    },
                )
            )
        tracer.update_trace(
            trace,
            output_text=tutor_response,
            metadata={
                "status": "success",
                "latency_ms": round(latency_seconds * 1000, 2),
                "tts_error": tts_error,
                "retrieval_count": retrieval.count,
                "retrieval_error": retrieval.error,
            },
        )
        tracer.flush()
        evaluation_service = EvaluationService(config)
        evaluation_record = evaluation_service.evaluate_interaction(
            request_id=request_id,
            session_id=session_id,
            input_text=user_text,
            output_text=tutor_response,
            expected_output=expected_output,
            reference_context=reference_context,
            model_name=config.ollama_model,
            provider=llm_metadata.provider if llm_metadata is not None else "ollama",
            task_type=task_type,
            tags=tags,
            latency_ms=round(latency_seconds * 1000, 2),
            input_tokens=llm_metadata.input_tokens if llm_metadata is not None else None,
            output_tokens=llm_metadata.output_tokens if llm_metadata is not None else None,
            token_source=llm_metadata.token_source if llm_metadata is not None else None,
            tool_calls=tool_calls,
            task_success=task_success,
            metadata={
                "mode": mode,
                "stt_model_name": stt_model_name,
                "tts_enabled": enable_tts,
                "tts_error": tts_error,
                "retrieval_count": retrieval.count,
                "retrieval_error": retrieval.error,
                "request_source": "api",
            },
        )
        evaluation_service.persist_interaction(evaluation_record)
        if config.metrics_enabled and config.prometheus_enabled:
            observe_evaluation_interaction(
                task_type=task_type,
                status=evaluation_record.workflow_status,
                success=evaluation_record.metrics.task_success_rate == 1.0,
            )
            for tool_call in tool_calls:
                observe_evaluation_tool_call(
                    tool_name=tool_call.name,
                    status=tool_call.status,
                    latency_seconds=(tool_call.latency_ms or 0.0) / 1000 if tool_call.latency_ms is not None else None,
                )
        log_event(
            logger,
            "chat_turn",
            config=config,
            model=config.ollama_model,
            mode=mode,
            task_type=task_type,
            stt_model_name=stt_model_name,
            tts_enabled=enable_tts,
            tts_error=tts_error,
            retrieval_count=retrieval.count,
            retrieval_error=retrieval.error,
            status="success",
            latency_ms=round(latency_seconds * 1000, 2),
        )

    return ChatResponse(
        request_id=request_id,
        session_id=session_id,
        mode=mode,
        model_name=config.ollama_model,
        voice=_voice_info(voice) if enable_tts else None,
        user_text=user_text,
        tutor_response=tutor_response,
        audio_url=audio_url,
        tts_error=tts_error,
        pronunciation_feedback=pronunciation_feedback,
        sources=[_rag_source_info(source) for source in retrieval.sources],
        retrieval_count=retrieval.count,
        retrieval_error=retrieval.error,
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
        stt_device=config.stt_device,
        stt_compute_type=config.stt_compute_type,
        piper_cuda=config.piper_cuda,
        vad=VadSettings(
            energy_threshold=config.vad_energy_threshold,
            silence_seconds=config.vad_silence_seconds,
            min_speech_seconds=config.vad_min_speech_seconds,
            max_seconds=config.vad_max_seconds,
            chunk_ms=config.vad_chunk_ms,
            sample_rate=config.sample_rate,
        ),
        rag=RagStatus(
            enabled=config.rag_enabled,
            vector_db=config.rag_vector_db,
            embedding_model=config.rag_embedding_model,
            collection=config.qdrant_collection,
            top_k=config.rag_top_k,
            score_threshold=config.rag_score_threshold,
            qdrant_url=config.qdrant_url,
            knowledge_dir=str(config.knowledge_dir),
        ),
    )


def _api_health(config: AppConfig) -> ApiHealthResponse:
    return ApiHealthResponse(
        status="ok",
        service=config.service_name,
        environment=config.observability_env,
        langfuse_enabled=config.langfuse_enabled,
        evidently_enabled=config.evidently_enabled,
        evaluation_enabled=config.evaluation_enabled,
        metrics_enabled=config.metrics_enabled,
        prometheus_enabled=config.prometheus_enabled,
        grafana_enabled=config.grafana_enabled,
    )


def _safe_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: object) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _latest_eval_run(config: AppConfig) -> ObservabilityRunSummary | None:
    results_dir = config.evaluation_data_dir / "results"
    if not results_dir.exists():
        return None

    candidates = sorted(results_dir.glob("eval_summary_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = _safe_json(path)
        if payload is None:
            continue
        try:
            return ObservabilityRunSummary.model_validate(payload)
        except Exception:
            continue
    return None


def _observability_summary(config: AppConfig) -> ObservabilitySummaryResponse:
    interactions_dir = config.evaluation_data_dir / "interactions"
    records: list[dict[str, object]] = []
    if interactions_dir.exists():
        for path in sorted(interactions_dir.glob("*.json")):
            payload = _safe_json(path)
            if payload is not None:
                records.append(payload)

    total_interactions = len(records)
    recent_cutoff = datetime.now(UTC).timestamp() - (24 * 60 * 60)
    interactions_last_24h = 0
    total_errors = 0
    latencies: list[float] = []
    feedback_scores: list[float] = []
    tool_call_count = 0
    tool_call_error_count = 0
    task_success_values: list[float] = []
    last_interaction_at: str | None = None

    for record in records:
        timestamp_value = record.get("timestamp")
        if isinstance(timestamp_value, str):
            if last_interaction_at is None or timestamp_value > last_interaction_at:
                last_interaction_at = timestamp_value
            try:
                if datetime.fromisoformat(timestamp_value).timestamp() >= recent_cutoff:
                    interactions_last_24h += 1
            except ValueError:
                pass

        metrics = record.get("metrics")
        metrics_dict = metrics if isinstance(metrics, dict) else {}
        latency = _float_or_none(record.get("latency_ms"))
        if latency is None:
            latency = _float_or_none(metrics_dict.get("latency_ms"))
        if latency is not None:
            latencies.append(latency)

        feedback_score = _float_or_none(metrics_dict.get("user_feedback_score"))
        if feedback_score is None:
            feedback_score = _float_or_none(record.get("user_feedback_score"))
        if feedback_score is not None:
            feedback_scores.append(feedback_score)

        task_success = _float_or_none(metrics_dict.get("task_success_rate"))
        if task_success is not None:
            task_success_values.append(task_success)

        error_message = record.get("error_message")
        if isinstance(error_message, str) and error_message:
            total_errors += 1

        tool_calls = record.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
            tool_call_error_count += sum(
                1
                for tool_call in tool_calls
                if isinstance(tool_call, dict) and tool_call.get("status") == "error"
            )
        else:
            tool_call_count += _int_or_zero(metrics_dict.get("tool_calls_count"))

    average_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    average_feedback_score = (
        round(sum(feedback_scores) / len(feedback_scores), 2) if feedback_scores else None
    )
    task_success_rate = (
        round(sum(task_success_values) / len(task_success_values), 4) if task_success_values else None
    )

    return ObservabilitySummaryResponse(
        status="ok",
        service=config.service_name,
        environment=config.observability_env,
        langfuse_enabled=config.langfuse_enabled,
        langfuse_url=config.langfuse_public_url if config.langfuse_enabled else None,
        evaluation_enabled=config.evaluation_enabled,
        metrics_enabled=config.metrics_enabled,
        prometheus_enabled=config.prometheus_enabled,
        rag_enabled=config.rag_enabled,
        total_interactions=total_interactions,
        interactions_last_24h=interactions_last_24h,
        total_errors=total_errors,
        average_latency_ms=average_latency_ms,
        average_feedback_score=average_feedback_score,
        tool_call_count=tool_call_count,
        tool_call_error_count=tool_call_error_count,
        task_success_rate=task_success_rate,
        last_interaction_at=last_interaction_at,
        latest_run=_latest_eval_run(config),
    )


@app.get("/api/health", response_model=ApiHealthResponse)
def get_api_health() -> ApiHealthResponse:
    return _api_health(_load_config_or_500())


@app.get("/api/observability/health", response_model=ObservabilityHealthResponse)
def get_observability_health() -> ObservabilityHealthResponse:
    config = _load_config_or_500()
    ollama_status = "ok"
    try:
        OllamaClient(config).list_models()
    except OllamaError:
        ollama_status = "unavailable"

    return ObservabilityHealthResponse(
        **_api_health(config).model_dump(),
        langfuse_configured=langfuse_credentials_configured(config),
        ollama_status=ollama_status,
        stt_device=config.stt_device,
        stt_compute_type=config.stt_compute_type,
        piper_cuda=config.piper_cuda,
    )


@app.get("/api/observability/summary", response_model=ObservabilitySummaryResponse)
def get_observability_summary() -> ObservabilitySummaryResponse:
    return _observability_summary(_load_config_or_500())


@app.get("/api/metrics")
def get_api_metrics():
    config = _load_config_or_500()
    return metrics_response(config.metrics_enabled and config.prometheus_enabled)


@app.get("/metrics")
def get_metrics():
    config = _load_config_or_500()
    return metrics_response(config.metrics_enabled and config.prometheus_enabled)


def _stt_engine_key(config: AppConfig) -> tuple[str, str, str, str]:
    return (
        config.stt_model_size,
        config.stt_language,
        config.stt_device,
        config.stt_compute_type,
    )


def _get_stt_engine(config: AppConfig) -> CachedSttEngine:
    key = _stt_engine_key(config)
    with stt_engines_lock:
        cached_engine = stt_engines.get(key)
        if cached_engine is None:
            cached_engine = CachedSttEngine(engine=SpeechToTextEngine(config), lock=Lock())
            stt_engines[key] = cached_engine
        return cached_engine


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
        expected_output=request.expected_output,
        reference_context=request.reference_context,
        task_type=request.task_type,
        tags=request.tags,
        task_success=request.task_success,
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

    stt_engine = _get_stt_engine(config)
    started_at = perf_counter()
    try:
        with stt_engine.lock:
            transcription: TranscriptionResult = stt_engine.engine.transcribe_detailed(audio_path)
    except SpeechToTextError as exc:
        latency_seconds = perf_counter() - started_at
        if config.metrics_enabled and config.prometheus_enabled:
            observe_stt_call(
                backend=stt_engine.engine.backend_name,
                status="error",
                latency_seconds=latency_seconds,
            )
        log_event(
            logger,
            "stt_call",
            config=config,
            session_id=selected_session_id,
            backend=stt_engine.engine.backend_name,
            device=config.stt_device,
            compute_type=config.stt_compute_type,
            status="error",
            latency_ms=round(latency_seconds * 1000, 2),
            error_message=str(exc),
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    latency_seconds = perf_counter() - started_at
    if config.metrics_enabled and config.prometheus_enabled:
        observe_stt_call(
            backend=stt_engine.engine.backend_name,
            status="success",
            latency_seconds=latency_seconds,
        )
    log_event(
        logger,
        "stt_call",
        config=config,
        session_id=selected_session_id,
        backend=stt_engine.engine.backend_name,
        device=config.stt_device,
        compute_type=config.stt_compute_type,
        status="success",
        latency_ms=round(latency_seconds * 1000, 2),
    )

    return _chat_response(
        config=config,
        session_id=selected_session_id,
        mode=parsed_mode,
        user_text=transcription.text,
        stt_model_name=stt_engine.engine.backend_name,
        enable_tts=enable_tts,
        pronunciation_feedback=transcription.pronunciation_feedback,
    )


@app.post("/api/feedback", response_model=FeedbackResponse)
def post_feedback(request: FeedbackRequest) -> FeedbackResponse:
    config = _load_config_or_500()
    path = EvaluationService(config).persist_feedback(
        request_id=request.request_id,
        score=request.score,
        comment=request.comment,
    )
    if path is not None and config.metrics_enabled and config.prometheus_enabled:
        observe_user_feedback(request.score)
    return FeedbackResponse(request_id=request.request_id, saved=path is not None)


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

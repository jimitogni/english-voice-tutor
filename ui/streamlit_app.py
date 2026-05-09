from __future__ import annotations

import base64
import binascii
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, ConfigError, load_config
from app.focus_words import FocusWordsError, FocusWordsStore
from app.llm_client import OllamaError, OllamaClient
from app.memory import ConversationMemory
from app.prompts import TutorMode, available_modes, get_mode_definition
from app.stt import SpeechToTextEngine, SpeechToTextError
from app.tts import TextToSpeechEngine, TextToSpeechError
from app.tutor_agent import EnglishTutorAgent
from app.utils import ensure_directory, file_timestamp
from app.voice_profiles import apply_voice_profile, voice_profile_for_model
from ui.silence_recorder import BrowserRecording, silence_recorder


RECOMMENDED_OLLAMA_MODELS = ["llama3.2:3b", "qwen3:4b", "gemma3:4b"]


def unique_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered_models: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            ordered_models.append(model)
    return ordered_models


def available_ollama_models(config: AppConfig) -> list[str]:
    try:
        return OllamaClient(config).list_models()
    except OllamaError as exc:
        st.warning(f"Could not list Ollama models: {exc}")
        return []


def get_agent(mode: TutorMode, config: AppConfig) -> EnglishTutorAgent:
    memory_key = f"memory_{mode}"
    if memory_key not in st.session_state:
        st.session_state[memory_key] = ConversationMemory(config)

    return EnglishTutorAgent(
        llm_client=OllamaClient(config),
        memory=st.session_state[memory_key],
        config=config,
        mode=mode,
    )


def render_history() -> None:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("model"):
                st.caption(f"LLM model: `{message['model']}`")
            if message["role"] == "assistant" and message.get("voice"):
                st.caption(f"TTS voice: {message['voice']}")
            st.write(message["content"])
            if message.get("audio_path"):
                render_audio(message["audio_path"], autoplay=False)


def render_audio(audio_path: str, autoplay: bool) -> None:
    path = Path(audio_path)
    if not path.exists():
        st.warning(f"Audio file is missing: {path}")
        return

    if not autoplay:
        st.audio(str(path))
        return

    audio_bytes = path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    audio_id = f"audio_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    components.html(
        f"""
        <audio id="{audio_id}" controls autoplay style="width: 100%;">
          <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        </audio>
        <script>
          const audio = document.getElementById("{audio_id}");
          audio.play().catch(() => {{
            const note = document.createElement("div");
            note.style.fontSize = "0.85rem";
            note.style.color = "#666";
            note.style.marginTop = "0.25rem";
            note.textContent = "Autoplay was blocked by the browser. Press play to hear the answer.";
            audio.parentNode.appendChild(note);
          }});
        </script>
        """,
        height=92,
    )


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), minimum), maximum)


def browser_recording_suffix(recording: BrowserRecording) -> str:
    name = recording.get("name")
    if name:
        suffix = Path(str(name)).suffix.lower()
        if suffix in {".wav", ".webm", ".ogg", ".m4a", ".mp3", ".mp4"}:
            return suffix

    mime_type = recording.get("mime_type", "").lower()
    if "ogg" in mime_type:
        return ".ogg"
    if "mp4" in mime_type:
        return ".m4a"
    if "mpeg" in mime_type:
        return ".mp3"
    if "wav" in mime_type:
        return ".wav"
    return ".webm"


def transcribe_saved_audio(audio_path: Path) -> str | None:
    try:
        result = SpeechToTextEngine(load_config()).transcribe_detailed(audio_path)
    except SpeechToTextError as exc:
        st.error(str(exc))
        return None

    st.caption(f"Transcribed from `{audio_path.name}`")
    if result.pronunciation_feedback:
        st.info(result.pronunciation_feedback)
    return result.text


def transcribe_audio_file(audio_file, source_label: str) -> str | None:
    config = load_config()
    input_dir = ensure_directory(config.audio_inputs_dir)
    suffix = Path(audio_file.name).suffix or ".wav"
    audio_path = input_dir / f"{source_label}_{file_timestamp()}{suffix}"
    audio_path.write_bytes(audio_file.getbuffer())
    return transcribe_saved_audio(audio_path)


def transcribe_browser_recording(recording: BrowserRecording) -> str | None:
    data_url = recording.get("data_url")
    if not data_url or "," not in data_url:
        st.error("The browser recording did not include readable audio data.")
        return None

    try:
        _header, encoded_audio = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded_audio, validate=True)
    except (ValueError, binascii.Error):
        st.error("Could not decode the browser recording.")
        return None

    if not audio_bytes:
        st.warning("The browser returned an empty recording.")
        return None

    config = load_config()
    input_dir = ensure_directory(config.audio_inputs_dir)
    digest = hashlib.sha256(audio_bytes).hexdigest()[:8]
    suffix = browser_recording_suffix(recording)
    audio_path = input_dir / f"browser_vad_{file_timestamp()}_{digest}{suffix}"
    audio_path.write_bytes(audio_bytes)

    duration = recording.get("duration_seconds")
    if isinstance(duration, (int, float)):
        st.caption(f"Browser recording duration: {duration:.1f}s")

    return transcribe_saved_audio(audio_path)


def answer_with_agent(
    agent: EnglishTutorAgent,
    user_text: str,
    *,
    stream_response: bool,
    enable_tts: bool,
    autoplay_audio: bool,
    model_name: str,
    voice_label: str,
    config: AppConfig,
) -> tuple[str, str | None]:
    with st.chat_message("assistant"):
        st.caption(f"LLM model: `{model_name}`")
        if enable_tts:
            st.caption(f"TTS voice: {voice_label}")
        try:
            if stream_response:
                chunks: list[str] = []

                def chunk_iter():
                    for chunk in agent.reply_stream(user_text):
                        chunks.append(chunk)
                        yield chunk

                response = st.write_stream(chunk_iter())
                response_text = response if isinstance(response, str) else "".join(chunks)
            else:
                response_text = agent.reply(user_text)
                st.write(response_text)
        except OllamaError as exc:
            st.error(f"Ollama error: {exc}")
            return "", None

        audio_path = synthesize_for_browser(response_text, config) if enable_tts else None
        if audio_path:
            render_audio(audio_path, autoplay=autoplay_audio)

    return response_text, audio_path


def synthesize_for_browser(text: str, config: AppConfig) -> str | None:
    if config.tts_engine in {"none", "off", "disabled"}:
        return None

    try:
        return str(TextToSpeechEngine(config).synthesize(text))
    except TextToSpeechError as exc:
        st.warning(f"TTS unavailable: {exc}")
        return None


def render_focus_words_panel(config) -> None:
    st.header("Focus Words")
    store = FocusWordsStore(config)
    try:
        focus_words = store.list_words()
    except FocusWordsError as exc:
        st.error(str(exc))
        return

    st.caption(f"{len(focus_words)}/{config.focus_words_limit} saved")
    with st.form("add_focus_word", clear_on_submit=True):
        new_word = st.text_input("Add word or expression", placeholder="e.g. actually")
        submitted = st.form_submit_button("Add")
        if submitted:
            try:
                added = store.add_word(new_word)
            except FocusWordsError as exc:
                st.warning(str(exc))
            else:
                if not added:
                    st.info("That word is already in your focus list.")
                st.rerun()

    if not focus_words:
        st.caption("No focus words yet.")
        return

    for index, word in enumerate(focus_words):
        col_word, col_remove = st.columns([0.78, 0.22], vertical_alignment="center")
        col_word.write(f"{index + 1}. {word}")
        if col_remove.button("Remove", key=f"remove_focus_{hashlib.sha1(word.encode()).hexdigest()}"):
            try:
                store.remove_word(word)
            except FocusWordsError as exc:
                st.warning(str(exc))
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="English Voice Tutor")
    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        return

    st.title(config.assistant_name)
    st.caption(
        f"English Voice Tutor for {config.user_display_name} | "
        "Local Ollama + faster-whisper + Piper"
    )

    with st.sidebar:
        st.header("Session")
        st.write(f"Assistant: **{config.assistant_name}**")
        st.write(f"User: **{config.user_display_name}**")
        st.write(f"Context memory: last **{config.conversation_history_turns}** turns")
        render_focus_words_panel(config)
        mode = st.selectbox(
            "Tutor mode",
            options=available_modes(),
            format_func=lambda value: get_mode_definition(value).label,
        )
        installed_models = available_ollama_models(config)
        model_options = unique_models(
            [config.ollama_model, *RECOMMENDED_OLLAMA_MODELS, *installed_models]
        )
        selected_model = st.selectbox(
            "Ollama model",
            options=model_options,
            index=model_options.index(config.ollama_model)
            if config.ollama_model in model_options
            else 0,
        )
        if selected_model not in installed_models:
            st.warning(
                f"Model `{selected_model}` is not installed. "
                f"Run `ollama pull {selected_model}`."
            )
        config = replace(config, ollama_model=selected_model)
        selected_voice = voice_profile_for_model(config, config.ollama_model)
        config = apply_voice_profile(config, selected_voice)
        st.write(f"TTS voice: **{selected_voice.label}**")
        if not selected_voice.is_available:
            st.warning(
                f"TTS voice files for {selected_voice.label} are missing in `models/piper/`."
            )

        stream_response = st.toggle("Stream LLM response", value=config.llm_stream)
        enable_tts = st.toggle("Generate browser audio", value=config.tts_engine == "piper")
        autoplay_audio = st.toggle(
            "Autoplay answer audio",
            value=True,
            disabled=not enable_tts,
        )
        with st.expander("Voice detection", expanded=False):
            st.caption(
                "If Jarvis cuts you off, increase the pause time. "
                "If your quiet words are missed, lower the threshold."
            )
            vad_silence_seconds = st.slider(
                "Pause before auto-send",
                min_value=0.5,
                max_value=5.0,
                value=clamp_float(config.vad_silence_seconds, 0.5, 5.0),
                step=0.1,
                format="%.1f s",
            )
            vad_max_seconds = st.slider(
                "Maximum recording length",
                min_value=5.0,
                max_value=90.0,
                value=clamp_float(config.vad_max_seconds, 5.0, 90.0),
                step=1.0,
                format="%.0f s",
            )
            vad_energy_threshold = st.slider(
                "Voice threshold",
                min_value=0.005,
                max_value=0.08,
                value=clamp_float(config.vad_energy_threshold, 0.005, 0.08),
                step=0.005,
                format="%.3f",
                help="Lower is more sensitive to quiet speech. Higher ignores more background noise.",
            )
            vad_min_speech_seconds = st.slider(
                "Minimum speech before auto-stop",
                min_value=0.1,
                max_value=2.0,
                value=clamp_float(config.vad_min_speech_seconds, 0.1, 2.0),
                step=0.1,
                format="%.1f s",
            )

        if st.button("Reset chat"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.info(
        f"Current LLM model: `{config.ollama_model}` | "
        f"TTS voice: **{selected_voice.label}**"
    )

    st.subheader("Voice")
    browser_recording = silence_recorder(
        sample_rate=config.sample_rate,
        energy_threshold=vad_energy_threshold,
        silence_seconds=vad_silence_seconds,
        min_speech_seconds=vad_min_speech_seconds,
        max_seconds=vad_max_seconds,
        key="browser_vad_recorder",
    )
    st.caption(
        "The browser recorder stops after silence, then sends the audio automatically."
    )
    uploaded_file = st.file_uploader(
        "Or upload WAV/MP3/M4A/WEBM input",
        type=["wav", "mp3", "m4a", "webm", "ogg"],
    )

    if browser_recording is not None:
        recording_id = browser_recording.get("id")
        if recording_id and st.session_state.get("last_browser_recording_id") != recording_id:
            st.session_state["last_browser_recording_id"] = recording_id
            if browser_recording.get("error"):
                st.warning(browser_recording["error"])
            else:
                with st.spinner("Transcribing your recording..."):
                    spoken_text = transcribe_browser_recording(browser_recording)
                if spoken_text:
                    st.session_state["pending_text"] = spoken_text

    if uploaded_file is not None and st.button("Transcribe uploaded audio"):
        uploaded_text = transcribe_audio_file(uploaded_file, "upload")
        if uploaded_text:
            st.session_state["pending_text"] = uploaded_text

    st.session_state.setdefault("messages", [])
    render_history()

    pending_text = st.session_state.pop("pending_text", None)
    user_text = st.chat_input("Type a message, record your voice, or upload audio")
    if pending_text and not user_text:
        user_text = pending_text

    if not user_text:
        return

    agent = get_agent(mode, config)
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    response_text, audio_path = answer_with_agent(
        agent,
        user_text,
        stream_response=stream_response,
        enable_tts=enable_tts,
        autoplay_audio=autoplay_audio,
        model_name=config.ollama_model,
        voice_label=selected_voice.label,
        config=config,
    )
    if not response_text:
        return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "audio_path": audio_path,
            "model": config.ollama_model,
            "voice": selected_voice.label if enable_tts else None,
        }
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import ConfigError, load_config
from app.llm_client import OllamaError, OllamaClient
from app.memory import ConversationMemory
from app.prompts import available_modes, get_mode_definition
from app.stt import SpeechToTextEngine, SpeechToTextError
from app.tts import TextToSpeechEngine, TextToSpeechError
from app.tutor_agent import EnglishTutorAgent
from app.utils import ensure_directory, file_timestamp


def get_agent(mode: str) -> EnglishTutorAgent:
    config = load_config()
    state_key = f"agent_{mode}"
    if state_key not in st.session_state:
        st.session_state[state_key] = EnglishTutorAgent(
            llm_client=OllamaClient(config),
            memory=ConversationMemory(config),
            config=config,
            mode=mode,  # type: ignore[arg-type]
        )
    return st.session_state[state_key]


def render_history() -> None:
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("audio_path"):
                st.audio(message["audio_path"])


def transcribe_upload(uploaded_file) -> str | None:
    config = load_config()
    input_dir = ensure_directory(config.audio_inputs_dir)
    suffix = Path(uploaded_file.name).suffix or ".wav"
    audio_path = input_dir / f"upload_{file_timestamp()}{suffix}"
    audio_path.write_bytes(uploaded_file.getbuffer())

    try:
        result = SpeechToTextEngine(config).transcribe_detailed(audio_path)
    except SpeechToTextError as exc:
        st.error(str(exc))
        return None

    st.caption(f"Transcribed from `{audio_path.name}`")
    if result.pronunciation_feedback:
        st.info(result.pronunciation_feedback)
    return result.text


def synthesize_for_browser(text: str) -> str | None:
    config = load_config()
    if config.tts_engine in {"none", "off", "disabled"}:
        return None

    try:
        return str(TextToSpeechEngine(config).synthesize(text))
    except TextToSpeechError as exc:
        st.warning(f"TTS unavailable: {exc}")
        return None


def main() -> None:
    st.set_page_config(page_title="English Voice Tutor")
    st.title("English Voice Tutor")
    st.caption("Local Ollama + faster-whisper + Piper")

    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        return

    with st.sidebar:
        st.header("Session")
        mode = st.selectbox(
            "Tutor mode",
            options=available_modes(),
            format_func=lambda value: get_mode_definition(value).label,
        )
        stream_response = st.toggle("Stream LLM response", value=config.llm_stream)
        enable_tts = st.toggle("Generate browser audio", value=config.tts_engine == "piper")
        uploaded_file = st.file_uploader("Optional WAV/MP3/M4A input", type=["wav", "mp3", "m4a"])

        if st.button("Reset chat"):
            for key in list(st.session_state.keys()):
                if key.startswith("agent_") or key == "messages":
                    del st.session_state[key]
            st.rerun()

    st.session_state.setdefault("messages", [])
    render_history()

    uploaded_text = None
    if uploaded_file is not None and st.button("Transcribe uploaded audio"):
        uploaded_text = transcribe_upload(uploaded_file)
        if uploaded_text:
            st.session_state["pending_text"] = uploaded_text

    pending_text = st.session_state.pop("pending_text", None)
    user_text = st.chat_input("Type a message or transcribe uploaded audio first")
    if pending_text and not user_text:
        user_text = pending_text

    if not user_text:
        return

    agent = get_agent(mode)
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
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
            return

        audio_path = synthesize_for_browser(response_text) if enable_tts else None
        if audio_path:
            st.audio(audio_path)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "audio_path": audio_path}
    )


if __name__ == "__main__":
    main()

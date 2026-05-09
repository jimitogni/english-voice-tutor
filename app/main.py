from __future__ import annotations

import argparse
import sys
from typing import cast

from app.audio_recorder import AudioRecorder, AudioRecordingError
from app.config import AppConfig, ConfigError, load_config
from app.focus_words import FocusWordsError, FocusWordsStore
from app.llm_client import OllamaError, OllamaModelNotFoundError, OllamaClient
from app.memory import ConversationMemory, ConversationMemoryError
from app.prompts import TutorMode, available_modes, get_mode_definition, mode_choices
from app.stt import SpeechToTextEngine, SpeechToTextError
from app.tutor_agent import EnglishTutorAgent
from app.tts import TextToSpeechEngine, TextToSpeechError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Terminal loop for a local English conversation tutor."
    )
    parser.add_argument(
        "--mode",
        choices=mode_choices(),
        default="choose",
        help="Tutor behavior mode. Use `choose` for an interactive menu.",
    )
    parser.add_argument(
        "--input",
        choices=("voice", "typed"),
        default="voice",
        help="Use microphone input or typed input. Use typed for debugging.",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip startup model availability check and let the chat request fail if needed.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=None,
        help="Override RECORD_SECONDS for voice input.",
    )
    parser.add_argument(
        "--record-mode",
        choices=("fixed", "vad"),
        default=None,
        help="Override RECORD_MODE. Use `vad` to stop after detected silence.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming LLM responses.",
    )
    parser.add_argument(
        "--no-stream-tts",
        action="store_true",
        help="Disable sentence-by-sentence TTS while keeping normal TTS enabled.",
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Print tutor responses without generating or playing speech.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the conversation session to data/conversations.",
    )
    return parser.parse_args(argv)


def choose_mode(requested_mode: str) -> TutorMode:
    if requested_mode != "choose":
        return cast(TutorMode, requested_mode)

    modes = available_modes()
    print("\nChoose a tutor mode:")
    for index, mode in enumerate(modes, start=1):
        definition = get_mode_definition(mode)
        print(f"  {index}. {definition.label} - {definition.description}")

    while True:
        try:
            choice = input("Mode number or name [1]: ").strip().lower()
        except EOFError:
            print("\nNo mode selected. Using Free Conversation.")
            return "free"

        if not choice:
            return "free"

        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(modes):
                return modes[index - 1]

        for mode in modes:
            definition = get_mode_definition(mode)
            if choice in {mode, definition.label.lower()}:
                return mode

        valid = ", ".join(str(index) for index in range(1, len(modes) + 1))
        names = ", ".join(modes)
        print(f"Please choose {valid} or one of: {names}.")


def check_ollama_ready(client: OllamaClient, configured_model: str) -> bool:
    try:
        client.ensure_model_available(configured_model)
    except OllamaModelNotFoundError as exc:
        print(f"\nModel unavailable: {exc}")
        return False
    except OllamaError as exc:
        print(f"\nOllama is not ready: {exc}")
        return False
    return True


def build_agent(config: AppConfig, mode: TutorMode) -> tuple[EnglishTutorAgent, OllamaClient]:
    client = OllamaClient(config)
    memory = ConversationMemory(config)
    agent = EnglishTutorAgent(
        llm_client=client,
        memory=memory,
        config=config,
        mode=mode,
    )
    return agent, client


def build_tts_engine(config: AppConfig, disable_tts: bool) -> TextToSpeechEngine | None:
    if disable_tts or config.tts_engine.strip().lower() in {"none", "off", "disabled"}:
        return None
    return TextToSpeechEngine(config)


def speak_or_continue(tts_engine: TextToSpeechEngine | None, text: str) -> bool:
    if tts_engine is None:
        return False

    try:
        output_path = tts_engine.speak(text)
    except TextToSpeechError as exc:
        print(f"TTS unavailable: {exc}")
        print("Continuing in text-only mode for this session.\n")
        return False

    print(f"Spoken audio: {output_path}")
    return True


def get_tutor_response(
    agent: EnglishTutorAgent,
    user_text: str,
    stt_model_name: str,
    *,
    stream_response: bool,
    tts_engine: TextToSpeechEngine | None,
    stream_tts: bool,
) -> tuple[str, TextToSpeechEngine | None]:
    if not stream_response:
        response = agent.reply(user_text, stt_model_name=stt_model_name)
        print(f"\nTutor: {response}\n")
        if tts_engine is not None and not speak_or_continue(tts_engine, response):
            tts_engine = None
        return response, tts_engine

    chunks: list[str] = []
    raw_chunk_iter = agent.reply_stream(user_text, stt_model_name=stt_model_name)

    def printing_chunks():
        for chunk in raw_chunk_iter:
            chunks.append(chunk)
            print(chunk, end="", flush=True)
            yield chunk

    print("\nTutor: ", end="", flush=True)
    chunk_iter = printing_chunks()

    if tts_engine is not None and stream_tts:
        try:
            for _output_path in tts_engine.speak_stream(chunk_iter):
                pass
        except TextToSpeechError as exc:
            print(f"\nTTS unavailable: {exc}")
            print("Continuing in text-only mode for this session.\n")
            tts_engine = None
            for _chunk in raw_chunk_iter:
                chunks.append(_chunk)
                print(_chunk, end="", flush=True)
    else:
        for _chunk in chunk_iter:
            pass

    print("\n")
    response = "".join(chunks).strip()
    if tts_engine is not None and not stream_tts and not speak_or_continue(tts_engine, response):
        tts_engine = None
    return response, tts_engine


def print_session_header(
    *,
    input_mode: str,
    config: AppConfig,
    model: str,
    assistant_name: str,
    user_display_name: str,
    conversation_history_turns: int,
    mode: TutorMode,
    tts_engine: TextToSpeechEngine | None,
    tts_name: str,
    stream_response: bool,
    stream_tts: bool,
    record_mode: str | None = None,
    record_seconds: float | None = None,
) -> None:
    definition = get_mode_definition(mode)
    print(f"\nEnglish Voice Tutor - {input_mode} mode")
    print(f"Assistant: {assistant_name}")
    print(f"User: {user_display_name}")
    print(f"Model: {model}")
    print(f"Context memory: last {conversation_history_turns} turns")
    try:
        focus_count = len(FocusWordsStore(config).list_words())
    except FocusWordsError:
        focus_count = 0
    print(f"Focus words: {focus_count}")
    print(f"Tutor mode: {definition.label}")
    print(f"Focus: {definition.description}")
    print(f"TTS: {'off' if tts_engine is None else tts_name}")
    print(f"LLM streaming: {'on' if stream_response else 'off'}")
    if tts_engine is not None:
        print(f"Streaming TTS: {'on' if stream_tts else 'off'}")
    if record_mode is not None:
        print(f"Recording mode: {record_mode}")
    if record_seconds is not None:
        print(f"Recording duration: {record_seconds:g} seconds")


def play_starter_if_needed(
    agent: EnglishTutorAgent,
    tts_engine: TextToSpeechEngine | None,
) -> TextToSpeechEngine | None:
    try:
        starter_response = agent.start_session()
    except OllamaError as exc:
        print(f"\nOllama error while starting the mode: {exc}\n")
        return tts_engine

    if starter_response is None:
        return tts_engine

    print(f"\nTutor: {starter_response}\n")
    if tts_engine is not None and not speak_or_continue(tts_engine, starter_response):
        return None
    return tts_engine


def save_conversation_if_requested(
    agent: EnglishTutorAgent,
    config: AppConfig,
    save_session: bool,
) -> None:
    if not save_session or not config.save_conversations:
        return

    try:
        saved_path = agent.memory.save_session()
    except ConversationMemoryError as exc:
        print(f"Conversation was not saved: {exc}")
        return

    if saved_path is not None:
        print(f"Conversation saved to {saved_path}")


def run_typed_loop(
    config: AppConfig,
    mode: TutorMode,
    skip_model_check: bool,
    save_session: bool,
    disable_tts: bool,
    stream_response: bool,
    stream_tts: bool,
) -> int:
    agent, client = build_agent(config, mode)
    if not skip_model_check and not check_ollama_ready(client, config.ollama_model):
        print("\nHelpful commands:")
        print("  ollama serve")
        print(f"  ollama pull {config.ollama_model}")
        return 1

    tts_engine = build_tts_engine(config, disable_tts)

    print_session_header(
        input_mode="typed",
        config=config,
        model=config.ollama_model,
        assistant_name=config.assistant_name,
        user_display_name=config.user_display_name,
        conversation_history_turns=config.conversation_history_turns,
        mode=mode,
        tts_engine=tts_engine,
        tts_name=config.tts_engine,
        stream_response=stream_response,
        stream_tts=stream_tts,
    )
    print("Type /quit to exit, /help for commands.\n")
    tts_engine = play_starter_if_needed(agent, tts_engine)

    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                continue

            command = user_text.lower()
            if command in {"/quit", "/exit", "quit", "exit"}:
                break
            if command == "/help":
                print("Commands: /help, /quit")
                continue

            try:
                _response, tts_engine = get_tutor_response(
                    agent,
                    user_text,
                    "typed-input",
                    stream_response=stream_response,
                    tts_engine=tts_engine,
                    stream_tts=stream_tts,
                )
            except OllamaError as exc:
                print(f"\nOllama error: {exc}\n")
                continue
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except EOFError:
        print("\nInput closed.")
    finally:
        save_conversation_if_requested(agent, config, save_session)

    return 0


def run_voice_loop(
    config: AppConfig,
    mode: TutorMode,
    skip_model_check: bool,
    save_session: bool,
    disable_tts: bool,
    stream_response: bool,
    stream_tts: bool,
    record_seconds: float | None = None,
    record_mode: str | None = None,
) -> int:
    agent, client = build_agent(config, mode)
    if not skip_model_check and not check_ollama_ready(client, config.ollama_model):
        print("\nHelpful commands:")
        print("  ollama serve")
        print(f"  ollama pull {config.ollama_model}")
        return 1

    recorder = AudioRecorder(config)
    stt_engine = SpeechToTextEngine(config)
    tts_engine = build_tts_engine(config, disable_tts)
    duration = record_seconds or config.record_seconds
    selected_record_mode = record_mode or config.record_mode

    print_session_header(
        input_mode="voice",
        config=config,
        model=config.ollama_model,
        assistant_name=config.assistant_name,
        user_display_name=config.user_display_name,
        conversation_history_turns=config.conversation_history_turns,
        mode=mode,
        tts_engine=tts_engine,
        tts_name=config.tts_engine,
        stream_response=stream_response,
        stream_tts=stream_tts,
        record_mode=selected_record_mode,
        record_seconds=duration,
    )
    print("Press Enter to record, or type /quit to exit.\n")
    tts_engine = play_starter_if_needed(agent, tts_engine)

    try:
        while True:
            command = input("Ready: ").strip().lower()
            if command in {"/quit", "/exit", "quit", "exit"}:
                break
            if command == "/help":
                print("Commands: /help, /quit. Press Enter with no text to record.")
                continue
            if command:
                print("Press Enter with no text to record, or use `--input typed` for typing.")
                continue

            try:
                if selected_record_mode == "vad":
                    print(f"Recording with VAD for up to {duration:g} seconds...")
                else:
                    print(f"Recording for {duration:g} seconds...")

                audio_path = recorder.record(duration, mode=selected_record_mode)
                print(f"Saved audio: {audio_path}")

                print("Transcribing...")
                transcription = stt_engine.transcribe_detailed(audio_path)
                user_text = transcription.text
                print(f"You said: {user_text}")
                if transcription.pronunciation_feedback:
                    print(transcription.pronunciation_feedback)

                _response, tts_engine = get_tutor_response(
                    agent,
                    user_text,
                    stt_engine.backend_name,
                    stream_response=stream_response,
                    tts_engine=tts_engine,
                    stream_tts=stream_tts,
                )
            except AudioRecordingError as exc:
                print(f"\nAudio recording error: {exc}")
                print("Try `python -m app.main --input typed` while you fix microphone setup.\n")
                return 1
            except SpeechToTextError as exc:
                print(f"\nSpeech-to-text error: {exc}")
                print("Try another recording or use `python -m app.main --input typed`.\n")
                continue
            except OllamaError as exc:
                print(f"\nOllama error: {exc}\n")
                continue
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except EOFError:
        print("\nInput closed.")
    finally:
        save_conversation_if_requested(agent, config, save_session)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return 2

    mode = choose_mode(args.mode)
    stream_response = config.llm_stream and not args.no_stream
    stream_tts = config.stream_tts and not args.no_stream_tts

    if args.input == "typed":
        return run_typed_loop(
            config=config,
            mode=mode,
            skip_model_check=args.skip_model_check,
            save_session=not args.no_save,
            disable_tts=args.no_tts,
            stream_response=stream_response,
            stream_tts=stream_tts,
        )

    return run_voice_loop(
        config=config,
        mode=mode,
        skip_model_check=args.skip_model_check,
        save_session=not args.no_save,
        disable_tts=args.no_tts,
        stream_response=stream_response,
        stream_tts=stream_tts,
        record_seconds=args.record_seconds,
        record_mode=args.record_mode,
    )


if __name__ == "__main__":
    sys.exit(main())

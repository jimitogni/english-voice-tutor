from __future__ import annotations

from collections.abc import Iterator

from app.config import AppConfig, load_config
from app.focus_words import FocusWordsError, FocusWordsStore
from app.llm_client import OllamaClient
from app.memory import ConversationMemory
from app.observability import current_trace
from app.observability.langfuse_client import get_langfuse_tracer
from app.observability.privacy import sanitize_messages
from app.prompts import TutorMode, get_starter_prompt, get_system_prompt
from app.rag import RagRetriever, RetrievalContext, format_retrieval_context


class EnglishTutorAgent:
    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        memory: ConversationMemory | None = None,
        config: AppConfig | None = None,
        mode: TutorMode = "free",
        rag_retriever: RagRetriever | None = None,
    ) -> None:
        self.config = config or load_config()
        self.llm_client = llm_client or OllamaClient(self.config)
        self.memory = memory or ConversationMemory(self.config)
        self.mode = mode
        self.rag_retriever = rag_retriever or (
            RagRetriever(self.config) if self.config.rag_enabled else None
        )
        self.last_retrieval = RetrievalContext(
            sources=[],
            vector_db=self.config.rag_vector_db,
        )

    def _focus_words_prompt(self) -> str | None:
        try:
            focus_words = FocusWordsStore(self.config).list_words()
        except FocusWordsError:
            return None

        if not focus_words:
            return None

        words = ", ".join(f'"{word}"' for word in focus_words)
        return (
            "The user has chosen these focus words or expressions for extra practice: "
            f"{words}. Naturally include one or two when useful, ask the user to create "
            "sentences with them sometimes, and gently correct usage. Do not force every "
            "word into every answer."
        )

    def build_messages(self, user_text: str) -> list[dict[str, str]]:
        system_prompt = get_system_prompt(
            self.mode,
            assistant_name=self.config.assistant_name,
            user_display_name=self.config.user_display_name,
        )
        focus_words_prompt = self._focus_words_prompt()
        if focus_words_prompt:
            system_prompt = f"{system_prompt}\n\n{focus_words_prompt}"

        messages = [{"role": "system", "content": system_prompt}]
        self.last_retrieval = RetrievalContext(
            sources=[],
            vector_db=self.config.rag_vector_db,
        )
        if self.rag_retriever is not None:
            self.last_retrieval = self.rag_retriever.retrieve(user_text)
            retrieval_prompt = format_retrieval_context(
                self.last_retrieval.sources,
                max_chars=self.config.rag_context_char_limit,
            )
            if retrieval_prompt:
                messages.append({"role": "system", "content": retrieval_prompt})

        messages.extend(self.memory.chat_messages())
        messages.append({"role": "user", "content": user_text})
        return messages

    def reply(self, user_text: str, stt_model_name: str = "typed-input") -> str:
        messages = self.build_messages(user_text)
        get_langfuse_tracer().log_span(
            current_trace(),
            name="prompt_construction",
            output_value=sanitize_messages(messages, self.config),
            metadata={
                "mode": self.mode,
                "history_turns": len(self.memory.turns),
                "stt_model_name": stt_model_name,
                "retrieval_count": self.last_retrieval.count,
                "retrieval_error": self.last_retrieval.error,
            },
        )
        response = self.llm_client.chat(messages)
        self.memory.add_turn(
            user_text=user_text,
            tutor_response=response,
            model_name=self.config.ollama_model,
            stt_model_name=stt_model_name,
        )
        return response

    def reply_stream(
        self,
        user_text: str,
        stt_model_name: str = "typed-input",
    ) -> Iterator[str]:
        messages = self.build_messages(user_text)
        get_langfuse_tracer().log_span(
            current_trace(),
            name="prompt_construction",
            output_value=sanitize_messages(messages, self.config),
            metadata={
                "mode": self.mode,
                "history_turns": len(self.memory.turns),
                "stt_model_name": stt_model_name,
                "retrieval_count": self.last_retrieval.count,
                "retrieval_error": self.last_retrieval.error,
            },
        )
        chunks: list[str] = []

        for chunk in self.llm_client.chat_stream(messages):
            chunks.append(chunk)
            yield chunk

        response = "".join(chunks).strip()
        if response:
            self.memory.add_turn(
                user_text=user_text,
                tutor_response=response,
                model_name=self.config.ollama_model,
                stt_model_name=stt_model_name,
            )

    def start_session(self) -> str | None:
        starter_prompt = get_starter_prompt(self.mode)
        if starter_prompt is None:
            return None

        messages = [
            {
                "role": "system",
                "content": get_system_prompt(
                    self.mode,
                    assistant_name=self.config.assistant_name,
                    user_display_name=self.config.user_display_name,
                ),
            },
            {"role": "user", "content": starter_prompt},
        ]
        response = self.llm_client.chat(messages)
        self.memory.add_turn(
            user_text=f"[{self.mode} mode starter]",
            tutor_response=response,
            model_name=self.config.ollama_model,
            stt_model_name="mode-starter",
        )
        return response

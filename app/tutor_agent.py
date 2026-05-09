from __future__ import annotations

from app.config import AppConfig, load_config
from app.llm_client import OllamaClient
from app.memory import ConversationMemory
from app.prompts import TutorMode, get_starter_prompt, get_system_prompt


class EnglishTutorAgent:
    def __init__(
        self,
        llm_client: OllamaClient | None = None,
        memory: ConversationMemory | None = None,
        config: AppConfig | None = None,
        mode: TutorMode = "free",
    ) -> None:
        self.config = config or load_config()
        self.llm_client = llm_client or OllamaClient(self.config)
        self.memory = memory or ConversationMemory(self.config)
        self.mode = mode

    def build_messages(self, user_text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": get_system_prompt(self.mode)}]
        messages.extend(self.memory.chat_messages())
        messages.append({"role": "user", "content": user_text})
        return messages

    def reply(self, user_text: str, stt_model_name: str = "typed-input") -> str:
        messages = self.build_messages(user_text)
        response = self.llm_client.chat(messages)
        self.memory.add_turn(
            user_text=user_text,
            tutor_response=response,
            model_name=self.config.ollama_model,
            stt_model_name=stt_model_name,
        )
        return response

    def start_session(self) -> str | None:
        starter_prompt = get_starter_prompt(self.mode)
        if starter_prompt is None:
            return None

        messages = [
            {"role": "system", "content": get_system_prompt(self.mode)},
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

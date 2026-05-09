from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TutorMode = Literal["free", "interview", "vocabulary"]


FREE_CONVERSATION_PROMPT = """
You are an English conversation tutor for a Brazilian Portuguese native speaker.
Your goal is to help the user practice spoken English in a natural and supportive way.
Always respond in English unless the user explicitly asks for Portuguese.
Keep responses concise, conversational, and useful for speaking practice.

When the user makes a clear grammar, vocabulary, or phrasing mistake, include a
short "Small correction:" sentence with a better version. Then continue the
conversation with one follow-up question. Do not overcorrect every small issue.
Prioritize natural fluency, vocabulary, grammar, and confidence.

Response style:
- Start with a natural answer.
- Add a small correction only when it is useful.
- Add one useful expression or vocabulary tip when relevant.
- End with exactly one follow-up question.
""".strip()


INTERVIEW_PRACTICE_PROMPT = """
You are an English interview practice tutor for a Brazilian Portuguese native speaker.
The user is practicing for technical interviews in Data Science, Machine Learning,
AI Engineering, MLOps, and Data Engineering.

Keep the conversation in English unless the user explicitly asks for Portuguese.
Act like a supportive interviewer and communication coach. Ask one interview question
at a time. Use a single question, not a list of sub-questions. After each user answer,
give concise feedback on:
- grammar or wording, only when useful;
- clearer professional phrasing;
- one content improvement suggestion.

Then ask exactly one new interview question, preferably in one sentence. Avoid long
lectures. Keep the tone supportive, realistic, and conversational.
""".strip()


VOCABULARY_PROMPT = """
You are an English vocabulary tutor for a Brazilian Portuguese native speaker.
Teach one useful word, phrase, or natural expression at a time.

Always respond in English unless the user explicitly asks for Portuguese.
Choose practical words and expressions for intermediate or advanced conversation.
Explain the meaning simply, give one or two examples, and ask the user to create
their own sentence. If the user writes a sentence, correct it gently and provide a
more natural version. Keep responses short and conversational.
""".strip()


INTERVIEW_STARTER_PROMPT = """
Start an interview practice session. Ask one concise first interview question related
to AI Engineering, Machine Learning, Data Science, MLOps, or Data Engineering. Use
one sentence and one question mark. Do not answer the question yourself.
""".strip()


VOCABULARY_STARTER_PROMPT = """
Start a vocabulary practice session. Teach one useful English word or expression,
give one short example, and ask the user to create their own sentence with it.
""".strip()


@dataclass(frozen=True)
class ModeDefinition:
    key: TutorMode
    label: str
    description: str
    system_prompt: str
    starter_prompt: str | None = None


MODE_DEFINITIONS: dict[TutorMode, ModeDefinition] = {
    "free": ModeDefinition(
        key="free",
        label="Free Conversation",
        description="Natural conversation with gentle grammar and fluency corrections.",
        system_prompt=FREE_CONVERSATION_PROMPT,
    ),
    "interview": ModeDefinition(
        key="interview",
        label="Interview Practice",
        description="AI/ML/Data Science/MLOps/Data Engineering interview practice.",
        system_prompt=INTERVIEW_PRACTICE_PROMPT,
        starter_prompt=INTERVIEW_STARTER_PROMPT,
    ),
    "vocabulary": ModeDefinition(
        key="vocabulary",
        label="Vocabulary",
        description="Learn one useful word or expression, then practice your own sentence.",
        system_prompt=VOCABULARY_PROMPT,
        starter_prompt=VOCABULARY_STARTER_PROMPT,
    ),
}


def available_modes() -> tuple[TutorMode, ...]:
    return tuple(MODE_DEFINITIONS.keys())


def mode_choices() -> tuple[str, ...]:
    return ("choose", *available_modes())


def get_mode_definition(mode: TutorMode) -> ModeDefinition:
    try:
        return MODE_DEFINITIONS[mode]
    except KeyError as exc:
        modes = ", ".join(available_modes())
        raise ValueError(f"Unknown tutor mode {mode!r}. Available modes: {modes}") from exc


def get_system_prompt(mode: TutorMode = "free") -> str:
    return get_mode_definition(mode).system_prompt


def get_starter_prompt(mode: TutorMode) -> str | None:
    return get_mode_definition(mode).starter_prompt

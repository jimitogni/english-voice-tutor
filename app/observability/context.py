from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class ObservabilityContext:
    request_id: str
    endpoint: str
    session_id: str | None = None
    user_id_hash: str | None = None
    trace: Any | None = None


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_endpoint: ContextVar[str | None] = ContextVar("endpoint", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_user_id_hash: ContextVar[str | None] = ContextVar("user_id_hash", default=None)
_trace: ContextVar[Any | None] = ContextVar("trace", default=None)


def current_request_id() -> str | None:
    return _request_id.get()


def current_endpoint() -> str | None:
    return _endpoint.get()


def current_session_id() -> str | None:
    return _session_id.get()


def current_user_id_hash() -> str | None:
    return _user_id_hash.get()


def current_trace() -> Any | None:
    return _trace.get()


@contextmanager
def use_observability_context(context: ObservabilityContext) -> Iterator[None]:
    tokens: list[tuple[ContextVar[Any], Token[Any]]] = [
        (_request_id, _request_id.set(context.request_id)),
        (_endpoint, _endpoint.set(context.endpoint)),
        (_session_id, _session_id.set(context.session_id)),
        (_user_id_hash, _user_id_hash.set(context.user_id_hash)),
        (_trace, _trace.set(context.trace)),
    ]
    try:
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)

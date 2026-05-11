from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from app.config import AppConfig
from app.observability.context import current_endpoint, current_request_id, current_session_id


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
            }:
                continue
            payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(config: AppConfig) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(config.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    if config.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
    root_logger.addHandler(handler)


def log_event(
    logger: logging.Logger,
    event_type: str,
    *,
    config: AppConfig,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    payload = {
        "service": config.service_name,
        "environment": config.observability_env,
        "request_id": fields.pop("request_id", None) or current_request_id(),
        "session_id": fields.pop("session_id", None) or current_session_id(),
        "endpoint": fields.pop("endpoint", None) or current_endpoint(),
        "event_type": event_type,
        **fields,
    }
    logger.log(level, event_type, extra=payload)

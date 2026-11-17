"""Structured logging. JSON in production, human-readable locally."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Attribute names a `LogRecord` already owns. Two jobs: (1) `extra={...}` must never use
# one of them - logging raises KeyError ("Attempt to overwrite 'message'"), which once
# turned every error response into a 500; (2) the formatter skips them when copying extras
# into the JSON, since the useful ones are already written explicitly above. The probe
# record keeps the set correct across Python versions.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, including any `extra=` fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise one record to a single JSON line.

        Values that are not JSON-safe are stringified rather than raising: a logging call must
        never be the thing that takes a request down.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Everything the caller attached with `extra=` lives in the record's __dict__.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install a single stdout handler on the root logger and route uvicorn through it.

    Args:
        level: Standard level name ("INFO", "DEBUG", ...); case-insensitive.
        fmt: "json" for one object per line (production), anything else for plain text.

    Called from the app factory at import time, so it runs before uvicorn starts logging
    and no line ever escapes in a different format.
    """
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        )

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # uvicorn ships its own handlers (plain text); left alone, its lines would escape our JSON
    # format. Clearing them and letting records propagate to the root handler fixes that.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """Return the stdlib logger for `name`; here so modules need not import logging."""
    return logging.getLogger(name)

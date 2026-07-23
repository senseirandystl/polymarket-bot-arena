"""Structured logging for the arena process.

Design goals (safety first — this runs a live trading loop):

* **Opt-in JSON.** Set ``ARENA_LOG_JSON=1`` and every handler emits one JSON
  object per line (ts / level / logger / event + structured context), ready for
  ``jq`` or a log shipper. Leave it unset and the output is byte-identical to
  the classic ``%(asctime)s [%(name)s] %(levelname)s: %(message)s`` text the
  existing logs, ``bin/arena`` probes, and any human grep already rely on.
* **Additive context.** Call sites attach structured fields via
  :func:`log_event` under a single ``ctx`` key on the LogRecord. The text
  formatter ignores ``ctx`` entirely (so text output never changes); the JSON
  formatter flattens it into the object. There is no risk of colliding with
  reserved ``LogRecord`` attributes because everything lives under ``ctx``.

Nothing here changes *what* is logged in text mode — only adds a machine
-readable representation and a uniform helper for the four event classes the
arena cares about: decisions, trades, evolution, and errors.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# LogRecord attributes that are structural, not caller context. Everything the
# caller passes lives under record.ctx, so we only need this to stay defensive
# if some other code sets stray attributes.
_RESERVED = frozenset(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "ctx"}

TEXT_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


class JSONFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, + flattened ctx."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        ctx = getattr(record, "ctx", None)
        if isinstance(ctx, dict):
            for key, value in ctx.items():
                # Never let context shadow the structural keys above.
                payload[key if key not in payload else f"ctx_{key}"] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def json_enabled() -> bool:
    return os.environ.get("ARENA_LOG_JSON", "").strip() not in ("", "0", "false", "False")


def configure_logging(log_file: Path | str, level: int = logging.INFO) -> None:
    """Configure the root logger's handlers (console + file).

    JSON when ``ARENA_LOG_JSON`` is truthy, classic text otherwise. Idempotent —
    replaces any handlers a prior call installed rather than stacking them.
    """
    formatter: logging.Formatter = (
        JSONFormatter() if json_enabled() else logging.Formatter(TEXT_FORMAT)
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(stream)
    root.addHandler(file_handler)
    root.setLevel(level)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Log ``event`` with structured ``fields`` attached under ``ctx``.

    In text mode the fields are dropped and only ``event`` is shown (identical
    to a plain ``logger.log(level, event)``); in JSON mode the fields become
    top-level keys. Pass ``exc_info=True`` inside an ``except`` block to capture
    the traceback in both modes.
    """
    logger.log(level, event, extra={"ctx": fields}, exc_info=exc_info)

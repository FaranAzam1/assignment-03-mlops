"""Structured JSON logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SECRET_KEY_RE = re.compile(r"(pass(word)?|token|secret|api[_-]?key|authorization|credential)", re.I)
SECRET_VALUE_RE = re.compile(r"(Bearer\s+[A-Za-z0-9._\-]+|eyJ[A-Za-z0-9._\-]{20,})")
REDACTED = "***REDACTED***"
MAX_STR = 300

_lock = threading.Lock()
_logger = logging.getLogger("mlops_agent")


def redact(value: Any, key: str = "") -> Any:
    """Return a copy of value with secrets masked and long strings truncated."""
    if key and SECRET_KEY_RE.search(key):
        return REDACTED
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        text = SECRET_VALUE_RE.sub(REDACTED, value)
        return text if len(text) <= MAX_STR else text[:MAX_STR] + "...[truncated]"
    return value


def log_event(log_file: Path | None, event: str, **fields: Any) -> dict[str, Any]:
    """Write one JSON line to the log file and the stdlib logger. Returns the record."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event}
    record.update({k: redact(v, k) for k, v in fields.items()})
    line = json.dumps(record, default=str)
    _logger.info(line)
    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with _lock, log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:  # logging must never break a request
            _logger.warning("could not write log file %s", log_file)
    return record

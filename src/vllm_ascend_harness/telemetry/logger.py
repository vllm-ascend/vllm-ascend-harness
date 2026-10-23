"""Logging utilities.

Provides a project-wide logger factory with consistent formatting,
structured JSON logging, and optional file rotation.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        for key in ("run_id", "task_id", "metric", "score"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(
    name: str = "vllm_ascend_harness",
    level: int = logging.INFO,
    json_format: bool = False,
    log_file: str | Path | None = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name (hierarchical via dotted notation).
    level:
        Logging level (default ``INFO``).
    json_format:
        If ``True``, emit structured JSON log records instead of
        plain-text format.
    log_file:
        Optional file path.  When set, a :class:`~logging.handlers.
        RotatingFileHandler` is attached with the given *max_bytes*
        and *backup_count*.
    max_bytes:
        Maximum log file size before rotation (default 10 MiB).
    backup_count:
        Number of rotated backup files to keep (default 5).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter: logging.Formatter
        if json_format:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                str(path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger

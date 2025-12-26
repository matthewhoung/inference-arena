"""Structured JSON logging module.

This module provides JSON-formatted logging with request context tracking
for the detection service. Logs metadata only for efficient log
analysis and performance monitoring.

Author: Matthew Hong
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

# Context variable for thread-safe request ID tracking
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging.

    Formats log records as JSON objects with standardized fields:
    - timestamp: ISO format timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - logger: Logger name
    - message: Log message
    - request_id: Optional request context ID
    - endpoint, latency_ms, status_code, detections: Optional extra fields
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON string representation
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request context if available
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields (endpoint, latency_ms, status_code, detections, etc.)
        for key in ["endpoint", "latency_ms", "status_code", "detections", "port"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup JSON structured logging for the application.

    Configures the root logger with:
    - JSON formatter
    - StreamHandler to stdout
    - Specified log level

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)

"""
Structured logging configuration for RentShield AI Analysis Engine.

Uses structlog for JSON-formatted log output with request_id context.
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any, Optional
from uuid import uuid4

import structlog
from structlog.types import Processor

from app.config import get_settings

# Context variable for request ID tracking
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> str:
    """
    Get current request ID from context or generate a new one.
    
    Returns:
        str: Request ID for tracing.
        
    Example:
        >>> request_id = get_request_id()
        >>> print(request_id)
        'req-a1b2c3d4-e5f6-...'
    """
    current = request_id_context.get()
    if current is None:
        current = f"req-{uuid4()}"
        request_id_context.set(current)
    return current


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Set request ID in context.
    
    Args:
        request_id: Optional request ID. If None, generates a new one.
        
    Returns:
        str: The set request ID.
        
    Example:
        >>> rid = set_request_id("custom-request-123")
        >>> print(rid)
        'custom-request-123'
    """
    if request_id is None:
        request_id = f"req-{uuid4()}"
    request_id_context.set(request_id)
    return request_id


def clear_request_id() -> None:
    """Clear request ID from context (call at end of request)."""
    request_id_context.set(None)


def add_request_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """
    Structlog processor to add request_id to all log entries.
    
    Args:
        logger: Logger instance.
        method_name: Name of the log method called.
        event_dict: Dictionary of log event data.
        
    Returns:
        dict: Updated event dictionary with request_id.
    """
    request_id = request_id_context.get()
    if request_id is not None:
        event_dict["request_id"] = request_id
    return event_dict


def setup_logging() -> None:
    """
    Configure structlog for the application.
    
    Sets up JSON logging with timestamps, log levels, and request ID injection.
    Should be called once at application startup.
    
    Example:
        >>> setup_logging()
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    settings = get_settings()
    
    # Define processors for structlog
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_request_id,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Choose renderer based on format setting
    if settings.LOG_FORMAT == "json":
        final_processor: Processor = structlog.processors.JSONRenderer()
    else:
        final_processor = structlog.dev.ConsoleRenderer(colors=True)
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final_processor,
        ],
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger for a module.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        structlog.BoundLogger: Configured logger instance.
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing request", endpoint="/classify")
        >>> logger.error("Failed to process", error="timeout")
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for logging with additional context.
    
    Example:
        >>> with LogContext(operation="classify_issue", issue_id="123"):
        ...     logger.info("Starting classification")
        ...     # All logs within this block will include operation and issue_id
    """
    
    def __init__(self, **context: Any) -> None:
        self.context = context
        self.token: Optional[object] = None
    
    def __enter__(self) -> "LogContext":
        structlog.contextvars.clear_contextvars()
        for key, value in self.context.items():
            structlog.contextvars.bind_contextvars(**{key: value})
        return self
    
    def __exit__(self, *args: Any) -> None:
        structlog.contextvars.clear_contextvars()

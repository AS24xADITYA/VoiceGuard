"""Structured logging configuration.

JSON output via structlog, correlation-ID middleware for per-analysis tracing.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Context variable for the current correlation ID.
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Return the current correlation ID, or empty string if unset."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current async context."""
    _correlation_id.set(cid)


def _add_correlation_id(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that injects the correlation ID."""
    cid = get_correlation_id()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for JSON output."""
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer()
        if log_level == "DEBUG"
        else structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Quiet noisy third-party loggers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(name).setLevel(logging.WARNING)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that sets a correlation ID on each request.

    If the request provides an X-Correlation-ID header, it is used;
    otherwise a fresh UUID is generated.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        cid = request.headers.get("x-correlation-id", str(uuid.uuid4()))
        set_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

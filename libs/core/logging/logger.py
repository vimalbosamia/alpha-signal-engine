"""
Structured logging setup.

JSON output in production. Colorised in TTY/dev.
Every log record carries: timestamp, level, asset_class, symbol, mode.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Call once at application startup."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    for lib in ("alpaca", "binance", "urllib3", "asyncio", "httpx", "websockets"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    is_tty = sys.stdout.isatty()
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]
    processors.append(
        structlog.dev.ConsoleRenderer(colors=True)
        if is_tty
        else structlog.processors.JSONRenderer()
    )
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str, **ctx: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger pre-populated with context fields."""
    from libs.core.config import get_settings
    return structlog.get_logger(name).bind(
        mode=get_settings().agent_mode.value,
        **ctx,
    )

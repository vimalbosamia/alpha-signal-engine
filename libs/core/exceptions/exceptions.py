"""
Exception hierarchy for the signal agent.

Every exception carries context for structured logging and safe recovery.
"""
from __future__ import annotations


class SignalAgentError(Exception):
    """Root exception for all agent errors."""


# ── Data layer ────────────────────────────────────────────────────────────────

class DataError(SignalAgentError):
    """Base for all data-related errors."""


class DataQualityError(DataError):
    """Data fails quality checks — signal generation must be blocked."""
    def __init__(self, symbol: str, reason: str, details: dict | None = None) -> None:
        self.symbol = symbol
        self.reason = reason
        self.details = details or {}
        super().__init__(f"[{symbol}] Data quality: {reason}")


class StaleDataError(DataError):
    """Data is too old for the current mode."""
    def __init__(self, symbol: str, age_seconds: float) -> None:
        self.symbol = symbol
        self.age_seconds = age_seconds
        super().__init__(f"[{symbol}] Stale: {age_seconds:.0f}s old")


class ProviderError(DataError):
    """A market-data provider returned an error or is unavailable."""
    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


class ProviderDisconnectError(ProviderError):
    """WebSocket or connection was lost."""


class NoDataError(DataError):
    """Expected data was not available (empty response)."""


# ── Analysis layer ────────────────────────────────────────────────────────────

class AnalysisError(SignalAgentError):
    """Error during candle analysis."""


class InsufficientDataError(AnalysisError):
    """Not enough bars to run analysis."""
    def __init__(self, symbol: str, required: int, available: int) -> None:
        super().__init__(
            f"[{symbol}] Need {required} bars, have {available}"
        )


# ── Signal layer ──────────────────────────────────────────────────────────────

class SignalError(SignalAgentError):
    """Error during signal generation."""


class ConfigurationError(SignalAgentError):
    """Invalid or missing configuration."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from libs.core.models.domain import SignalOutput

# ── Counters ──────────────────────────────────────────────────────────────────

SIGNALS_GENERATED = Counter(
    "trading_signals_generated_total",
    "Total signals generated",
    ["action", "asset_class", "strategy"],
)
DATA_QUALITY_BLOCKED = Counter(
    "trading_data_quality_blocked_total",
    "Data quality blocks",
    ["symbol", "asset_class"],
)
PROVIDER_ERRORS = Counter(
    "trading_provider_errors_total",
    "Provider errors",
    ["provider"],
)
PROVIDER_DISCONNECTS = Counter(
    "trading_provider_disconnects_total",
    "Provider disconnects",
    ["provider"],
)

# ── Histograms ────────────────────────────────────────────────────────────────

SIGNAL_CONFIDENCE = Histogram(
    "trading_signal_confidence",
    "Signal confidence distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0],
)

# ── Gauges ────────────────────────────────────────────────────────────────────

ACTIVE_SIGNALS = Gauge(
    "trading_active_signals_count",
    "Current active (non-NO_TRADE) signals",
)
PROVIDER_HEALTH = Gauge(
    "trading_provider_health",
    "Provider health (1=healthy, 0=unhealthy)",
    ["provider"],
)


class MetricsCollector:
    def record_signal(self, signal: SignalOutput) -> None:
        SIGNALS_GENERATED.labels(
            action=signal.action.value,
            asset_class=signal.asset_class.value,
            strategy=signal.strategy_name,
        ).inc()
        SIGNAL_CONFIDENCE.observe(signal.confidence)
        if signal.action.value != "NO_TRADE":
            ACTIVE_SIGNALS.inc()

    def record_data_quality_block(self, symbol: str, asset_class: str) -> None:
        DATA_QUALITY_BLOCKED.labels(symbol=symbol, asset_class=asset_class).inc()

    def record_provider_error(self, provider: str) -> None:
        PROVIDER_ERRORS.labels(provider=provider).inc()

    def record_provider_disconnect(self, provider: str) -> None:
        PROVIDER_DISCONNECTS.labels(provider=provider).inc()

    def set_provider_health(self, provider: str, healthy: bool) -> None:
        PROVIDER_HEALTH.labels(provider=provider).set(1.0 if healthy else 0.0)

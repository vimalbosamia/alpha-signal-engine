from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
from prometheus_client import REGISTRY

from libs.core.models.domain import (
    AssetClass,
    ConfluenceBreakdown,
    DataQualityStatus,
    MarketRegime,
    SessionType,
    SignalAction,
    SignalOutput,
    Timeframe,
    TrendDirection,
)
from libs.monitoring.alerts import AlertManager
from libs.monitoring.metrics import (
    ACTIVE_SIGNALS,
    DATA_QUALITY_BLOCKED,
    PROVIDER_ERRORS,
    PROVIDER_HEALTH,
    SIGNALS_GENERATED,
    MetricsCollector,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_signal(action: SignalAction = SignalAction.BUY) -> SignalOutput:
    breakdown = ConfluenceBreakdown(
        pattern_score=0.8,
        structure_score=0.7,
        level_score=0.6,
        volume_score=0.7,
        regime_score=0.6,
        session_score=1.0,
        risk_score=0.7,
        data_quality_score=1.0,
        weighted_total=0.72,
        weights={},
        factor_notes={},
        blocked_reasons=[],
        warning_tags=[],
    )
    return SignalOutput(
        symbol="AAPL",
        asset_class=AssetClass.STOCK,
        strategy_name="test_strategy",
        timeframe=Timeframe.FIVE_MIN,
        action=action,
        confidence=0.72,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=100.0,
        entry_zone_high=100.5,
        stop_loss=98.0,
        take_profit_1=105.0,
        estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.REGULAR,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=breakdown,
        generated_at=datetime.utcnow(),
    )


def _get_counter_value(metric, **labels) -> float:
    """Read a prometheus Counter value for given label set."""
    label_values = tuple(labels[k] for k in sorted(labels))
    try:
        return metric.labels(**labels)._value.get()
    except Exception:
        return 0.0


def _get_gauge_value(metric, **labels) -> float:
    """Read a prometheus Gauge value for given label set."""
    try:
        return metric.labels(**labels)._value.get()
    except Exception:
        return 0.0


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_record_signal_increments_signals_generated():
    """MetricsCollector.record_signal increments SIGNALS_GENERATED counter."""
    collector = MetricsCollector()
    signal = make_signal(SignalAction.BUY)

    before = _get_counter_value(
        SIGNALS_GENERATED,
        action="BUY",
        asset_class="stock",
        strategy="test_strategy",
    )
    collector.record_signal(signal)
    after = _get_counter_value(
        SIGNALS_GENERATED,
        action="BUY",
        asset_class="stock",
        strategy="test_strategy",
    )
    assert after == before + 1.0


def test_record_data_quality_block_increments_counter():
    """record_data_quality_block increments DATA_QUALITY_BLOCKED counter."""
    collector = MetricsCollector()

    before = _get_counter_value(
        DATA_QUALITY_BLOCKED, symbol="TSLA", asset_class="stock"
    )
    collector.record_data_quality_block("TSLA", "stock")
    after = _get_counter_value(
        DATA_QUALITY_BLOCKED, symbol="TSLA", asset_class="stock"
    )
    assert after == before + 1.0


def test_record_provider_error_increments_counter():
    """record_provider_error increments PROVIDER_ERRORS counter."""
    collector = MetricsCollector()

    before = _get_counter_value(PROVIDER_ERRORS, provider="alpaca")
    collector.record_provider_error("alpaca")
    after = _get_counter_value(PROVIDER_ERRORS, provider="alpaca")
    assert after == before + 1.0


def test_set_provider_health_healthy():
    """set_provider_health sets gauge to 1.0 for healthy."""
    collector = MetricsCollector()
    collector.set_provider_health("binance", healthy=True)
    assert _get_gauge_value(PROVIDER_HEALTH, provider="binance") == 1.0


def test_set_provider_health_unhealthy():
    """set_provider_health sets gauge to 0.0 for unhealthy."""
    collector = MetricsCollector()
    collector.set_provider_health("binance", healthy=False)
    assert _get_gauge_value(PROVIDER_HEALTH, provider="binance") == 0.0


async def test_alert_send_no_raise_empty_webhook():
    """AlertManager._send does NOT raise when webhook is empty string."""
    manager = AlertManager(webhook_url="")
    # Should complete silently with no exception
    await manager._send("test message")


async def test_alert_stale_data_logs_without_raising():
    """AlertManager.stale_data logs without raising."""
    manager = AlertManager(webhook_url="")
    # Should not raise even without a real webhook
    await manager.stale_data("AAPL", 120.0)


async def test_alert_provider_disconnect_no_raise():
    """AlertManager.provider_disconnect does not raise."""
    manager = AlertManager(webhook_url="")
    await manager.provider_disconnect("alpaca")


async def test_alert_missing_bars_no_raise():
    """AlertManager.missing_bars does not raise."""
    manager = AlertManager(webhook_url="")
    await manager.missing_bars("BTCUSDT", 5)

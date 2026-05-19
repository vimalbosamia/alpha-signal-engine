"""
Unit tests for PortfolioGuard.

All signals are built from SignalOutput — no network calls.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

import pytest

from libs.monitoring.portfolio_guard import (
    GuardConfig,
    GuardDecision,
    PortfolioGuard,
    _ActiveSignal,
    reset_portfolio_guard,
    get_portfolio_guard,
)
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_signal(
    action: SignalAction = SignalAction.BUY,
    symbol: str = "BTC/USDT",
    asset_class: AssetClass = AssetClass.CRYPTO,
) -> SignalOutput:
    breakdown = ConfluenceBreakdown(
        pattern_score=0.7, structure_score=0.7, level_score=0.7,
        volume_score=0.7, regime_score=0.7, session_score=0.7,
        risk_score=0.7, data_quality_score=0.7, weighted_total=0.7,
    )
    return SignalOutput(
        signal_id=uuid4(),
        symbol=symbol,
        asset_class=asset_class,
        strategy_name="test_strategy",
        timeframe=Timeframe.FIFTEEN_MIN,
        action=action,
        confidence=0.7,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=100.0,
        entry_zone_high=101.0,
        stop_loss=97.0,
        take_profit_1=106.0,
        estimated_risk_reward=2.0,
        market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=breakdown,
    )


def _make_guard(**kwargs) -> PortfolioGuard:
    return PortfolioGuard(GuardConfig(**kwargs))


# ── GuardConfig ────────────────────────────────────────────────────────────────

class TestGuardConfig:
    def test_defaults(self):
        cfg = GuardConfig()
        assert cfg.max_concurrent_signals == 50
        assert cfg.max_per_symbol == 3
        assert cfg.max_same_direction == 25
        assert cfg.max_per_asset_class == 30
        assert cfg.max_age_hours == 24.0

    def test_custom_values(self):
        cfg = GuardConfig(max_concurrent_signals=3, max_per_symbol=2)
        assert cfg.max_concurrent_signals == 3
        assert cfg.max_per_symbol == 2


# ── GuardDecision ──────────────────────────────────────────────────────────────

class TestGuardDecision:
    def test_allowed(self):
        d = GuardDecision(allowed=True)
        assert d.allowed is True
        assert d.blocked is False
        assert d.reason == ""

    def test_blocked(self):
        d = GuardDecision(allowed=False, reason="TOO_MANY")
        assert d.allowed is False
        assert d.blocked is True
        assert d.reason == "TOO_MANY"


# ── is_allowed — basic rules ───────────────────────────────────────────────────

class TestIsAllowedBasic:
    def test_no_trade_always_blocked(self):
        guard = _make_guard()
        sig = _make_signal(action=SignalAction.NO_TRADE)
        decision = guard.is_allowed(sig)
        assert decision.blocked

    def test_first_signal_allowed(self):
        guard = _make_guard()
        sig = _make_signal()
        assert guard.is_allowed(sig).allowed

    def test_allows_up_to_limit(self):
        guard = _make_guard(max_concurrent_signals=3)
        for i in range(3):
            sig = _make_signal(symbol=f"COIN{i}/USDT")
            assert guard.is_allowed(sig).allowed
            guard.register(sig)

    def test_blocks_at_limit(self):
        guard = _make_guard(max_concurrent_signals=2)
        for i in range(2):
            sig = _make_signal(symbol=f"COIN{i}/USDT")
            guard.register(sig)
        new_sig = _make_signal(symbol="COIN99/USDT")
        decision = guard.is_allowed(new_sig)
        assert decision.blocked
        assert "GUARD_MAX_SIGNALS" in decision.reason


# ── Rule 2: per-symbol ─────────────────────────────────────────────────────────

class TestPerSymbolRule:
    def test_same_symbol_blocked_at_limit(self):
        guard = _make_guard(max_per_symbol=1)
        sig1 = _make_signal(symbol="BTC/USDT", action=SignalAction.BUY)
        guard.register(sig1)
        sig2 = _make_signal(symbol="BTC/USDT", action=SignalAction.SELL)
        decision = guard.is_allowed(sig2)
        assert decision.blocked
        assert "GUARD_MAX_PER_SYMBOL" in decision.reason

    def test_different_symbol_allowed(self):
        guard = _make_guard(max_per_symbol=1)
        sig1 = _make_signal(symbol="BTC/USDT")
        guard.register(sig1)
        sig2 = _make_signal(symbol="ETH/USDT")
        assert guard.is_allowed(sig2).allowed

    def test_two_per_symbol_allows_second(self):
        guard = _make_guard(max_per_symbol=2)
        sig1 = _make_signal(symbol="BTC/USDT", action=SignalAction.BUY)
        guard.register(sig1)
        sig2 = _make_signal(symbol="BTC/USDT", action=SignalAction.SELL)
        assert guard.is_allowed(sig2).allowed


# ── Rule 3: directional concentration ─────────────────────────────────────────

class TestDirectionalRule:
    def test_blocks_when_too_many_buys(self):
        guard = _make_guard(max_same_direction=2)
        for i in range(2):
            sig = _make_signal(symbol=f"COIN{i}/USDT", action=SignalAction.BUY)
            guard.register(sig)
        new_buy = _make_signal(symbol="COIN99/USDT", action=SignalAction.BUY)
        decision = guard.is_allowed(new_buy)
        assert decision.blocked
        assert "GUARD_DIRECTION" in decision.reason

    def test_sell_still_allowed_when_buy_limit_hit(self):
        guard = _make_guard(max_same_direction=2)
        for i in range(2):
            sig = _make_signal(symbol=f"COIN{i}/USDT", action=SignalAction.BUY)
            guard.register(sig)
        sell_sig = _make_signal(symbol="COIN99/USDT", action=SignalAction.SELL)
        assert guard.is_allowed(sell_sig).allowed


# ── Rule 4: per-asset-class ────────────────────────────────────────────────────

class TestAssetClassRule:
    def test_blocks_at_asset_class_limit(self):
        guard = _make_guard(max_per_asset_class=2)
        for i in range(2):
            sig = _make_signal(
                symbol=f"COIN{i}/USDT", asset_class=AssetClass.CRYPTO
            )
            guard.register(sig)
        new_sig = _make_signal(symbol="COIN99/USDT", asset_class=AssetClass.CRYPTO)
        decision = guard.is_allowed(new_sig)
        assert decision.blocked
        assert "GUARD_ASSET_CLASS" in decision.reason

    def test_stock_allowed_when_crypto_limit_hit(self):
        guard = _make_guard(max_per_asset_class=2)
        for i in range(2):
            sig = _make_signal(symbol=f"COIN{i}/USDT", asset_class=AssetClass.CRYPTO)
            guard.register(sig)
        stock_sig = _make_signal(symbol="AAPL", asset_class=AssetClass.STOCK)
        assert guard.is_allowed(stock_sig).allowed


# ── register / release ─────────────────────────────────────────────────────────

class TestRegisterRelease:
    def test_register_increases_active_count(self):
        guard = _make_guard()
        assert guard.active_count == 0
        sig = _make_signal()
        guard.register(sig)
        assert guard.active_count == 1

    def test_register_idempotent(self):
        guard = _make_guard()
        sig = _make_signal()
        guard.register(sig)
        guard.register(sig)   # second call is no-op
        assert guard.active_count == 1

    def test_no_trade_register_is_noop(self):
        guard = _make_guard()
        sig = _make_signal(action=SignalAction.NO_TRADE)
        guard.register(sig)
        assert guard.active_count == 0

    def test_release_decreases_count(self):
        guard = _make_guard()
        sig = _make_signal()
        guard.register(sig)
        removed = guard.release(sig.signal_id)
        assert removed is True
        assert guard.active_count == 0

    def test_release_unknown_id_returns_false(self):
        guard = _make_guard()
        assert guard.release(uuid4()) is False

    def test_release_idempotent(self):
        guard = _make_guard()
        sig = _make_signal()
        guard.register(sig)
        guard.release(sig.signal_id)
        assert guard.release(sig.signal_id) is False  # already gone

    def test_release_allows_new_signal_for_same_symbol(self):
        guard = _make_guard(max_per_symbol=1)
        sig1 = _make_signal(symbol="BTC/USDT")
        guard.register(sig1)
        guard.release(sig1.signal_id)
        sig2 = _make_signal(symbol="BTC/USDT")
        assert guard.is_allowed(sig2).allowed

    def test_release_by_symbol(self):
        guard = _make_guard(max_per_symbol=3)
        sigs = [_make_signal(symbol="BTC/USDT") for _ in range(3)]
        for s in sigs:
            guard.register(s)
        assert guard.active_count == 3
        removed = guard.release_by_symbol("BTC/USDT")
        assert removed == 3
        assert guard.active_count == 0

    def test_release_by_symbol_only_removes_that_symbol(self):
        guard = _make_guard(max_per_symbol=2)
        btc = _make_signal(symbol="BTC/USDT")
        eth = _make_signal(symbol="ETH/USDT")
        guard.register(btc)
        guard.register(eth)
        guard.release_by_symbol("BTC/USDT")
        assert guard.active_count == 1


# ── Auto-expiry ────────────────────────────────────────────────────────────────

class TestAutoExpiry:
    def test_old_signals_are_expired_before_check(self):
        guard = _make_guard(max_concurrent_signals=1, max_age_hours=0.0)
        sig1 = _make_signal(symbol="BTC/USDT")
        guard.register(sig1)
        # Manually age the registered signal
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)
        guard._signals[sig1.signal_id].accepted_at = old_time

        # Now is_allowed should see 0 active (expired) and allow new signal
        sig2 = _make_signal(symbol="ETH/USDT")
        assert guard.is_allowed(sig2).allowed

    def test_fresh_signals_not_expired(self):
        guard = _make_guard(max_concurrent_signals=1, max_age_hours=24.0)
        sig1 = _make_signal(symbol="BTC/USDT")
        guard.register(sig1)
        sig2 = _make_signal(symbol="ETH/USDT")
        # Should be blocked because sig1 is still fresh
        assert guard.is_allowed(sig2).blocked


# ── exposure_summary ──────────────────────────────────────────────────────────

class TestExposureSummary:
    def test_empty_summary(self):
        guard = _make_guard()
        summary = guard.exposure_summary()
        assert summary["total"] == 0
        assert summary["buy"] == 0
        assert summary["sell"] == 0

    def test_mixed_summary(self):
        guard = _make_guard(max_per_symbol=2)
        buy1 = _make_signal(symbol="BTC/USDT", action=SignalAction.BUY)
        buy2 = _make_signal(symbol="ETH/USDT", action=SignalAction.BUY)
        sell1 = _make_signal(
            symbol="AAPL", action=SignalAction.SELL, asset_class=AssetClass.STOCK
        )
        for s in (buy1, buy2, sell1):
            guard.register(s)

        summary = guard.exposure_summary()
        assert summary["total"] == 3
        assert summary["buy"] == 2
        assert summary["sell"] == 1
        assert summary["crypto"] == 2
        assert summary["stock"] == 1


# ── Singleton ──────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_returns_same_instance(self):
        g1 = get_portfolio_guard()
        g2 = get_portfolio_guard()
        assert g1 is g2

    def test_reset_returns_new_instance(self):
        g1 = get_portfolio_guard()
        g2 = reset_portfolio_guard()
        assert g1 is not g2

    def test_reset_with_config(self):
        cfg = GuardConfig(max_concurrent_signals=999)
        guard = reset_portfolio_guard(cfg)
        assert guard._config.max_concurrent_signals == 999


# ── active_signals list ────────────────────────────────────────────────────────

class TestActiveSignals:
    def test_returns_copy(self):
        guard = _make_guard()
        sig = _make_signal()
        guard.register(sig)
        active = guard.active_signals()
        assert len(active) == 1
        assert isinstance(active[0], _ActiveSignal)
        assert active[0].symbol == sig.symbol

    def test_empty_when_no_signals(self):
        guard = _make_guard()
        assert guard.active_signals() == []


# ── Thread safety (smoke) ──────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_register_does_not_corrupt_count(self):
        import threading

        guard = _make_guard(max_concurrent_signals=1000)
        errors: list[Exception] = []

        def register_one():
            try:
                sig = _make_signal(symbol=f"COIN-{uuid4()}/USDT")
                if guard.is_allowed(sig).allowed:
                    guard.register(sig)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=register_one) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert guard.active_count <= 50


# ── Portfolio Heat ─────────────────────────────────────────────────────────────

def _mock_settings_risk_pct(risk_pct: float):
    """Return a patch context that makes get_settings() yield the given risk_pct."""
    from unittest.mock import MagicMock
    mock_settings = MagicMock()
    mock_settings.risk.max_risk_per_signal_pct = risk_pct
    return patch("libs.monitoring.portfolio_guard.get_settings", return_value=mock_settings)


class TestPortfolioHeat:
    def test_heat_blocks_when_at_limit(self):
        # 9 signals at 1% each = 9% heat; next signal at 1% would exceed 10%
        # Use high direction/concurrent limits so only heat rule fires
        with _mock_settings_risk_pct(1.0):
            guard = _make_guard(
                max_portfolio_heat_pct=10.0,
                max_concurrent_signals=20,
                max_same_direction=20,
                max_per_asset_class=20,
            )
            for i in range(9):
                sig = _make_signal(symbol=f"COIN{i}/USDT")
                assert guard.is_allowed(sig).allowed
                guard.register(sig)
            new_sig = _make_signal(symbol="COIN99/USDT")
            decision = guard.is_allowed(new_sig)
            assert decision.blocked
            assert "GUARD_PORTFOLIO_HEAT" in decision.reason

    def test_heat_allows_when_below_limit(self):
        with _mock_settings_risk_pct(1.0):
            guard = _make_guard(
                max_portfolio_heat_pct=10.0,
                max_concurrent_signals=20,
                max_same_direction=20,
                max_per_asset_class=20,
            )
            for i in range(5):
                sig = _make_signal(symbol=f"COIN{i}/USDT")
                guard.register(sig)
            new_sig = _make_signal(symbol="COIN99/USDT")
            assert guard.is_allowed(new_sig).allowed

    def test_exposure_summary_includes_heat(self):
        with _mock_settings_risk_pct(1.0):
            guard = _make_guard()
            sig = _make_signal()
            guard.register(sig)
            summary = guard.exposure_summary()
            assert "portfolio_heat_pct" in summary


# ── Daily Loss Circuit Breaker ─────────────────────────────────────────────────

class TestDailyLossCircuitBreaker:
    def test_circuit_breaker_blocks_after_loss_limit(self):
        guard = _make_guard(max_daily_loss_pct=3.0)
        guard.record_loss(3.0)   # hit the limit
        sig = _make_signal()
        assert guard.is_allowed(sig).blocked
        assert "GUARD_DAILY_LOSS" in guard.is_allowed(sig).reason

    def test_circuit_breaker_allows_before_limit(self):
        guard = _make_guard(max_daily_loss_pct=3.0)
        guard.record_loss(1.5)
        sig = _make_signal()
        assert guard.is_allowed(sig).allowed

    def test_loss_resets_on_new_day(self):
        guard = _make_guard(max_daily_loss_pct=3.0)
        # Set loss for "yesterday"
        guard._daily_loss_pct = 5.0
        guard._loss_day = "2020-01-01"
        # Today's check should reset and allow
        sig = _make_signal()
        assert guard.is_allowed(sig).allowed

    def test_exposure_summary_includes_daily_loss(self):
        guard = _make_guard()
        guard.record_loss(1.0)
        summary = guard.exposure_summary()
        assert "daily_loss_pct" in summary

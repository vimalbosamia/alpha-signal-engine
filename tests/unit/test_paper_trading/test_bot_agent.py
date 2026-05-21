"""
Unit tests for libs/paper_trading/bot_agent.py.

Tests cover: name, signal filtering, trade opening, exit logic (TP1 / SL),
stats reporting, and phase determination.
"""
from __future__ import annotations

import uuid

import pytest

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
from libs.paper_trading.bot_agent import BotAgent


# ── Concrete test subclass ─────────────────────────────────────────────────────

class ConcreteBotForTest(BotAgent):
    """Minimal subclass: take signals with confidence >= 0.6."""

    NAME = "TestBot"

    def should_take_signal(self, signal: SignalOutput) -> bool:
        return signal.confidence >= 0.6


# ── Signal factory ─────────────────────────────────────────────────────────────

_CONFLUENCE = ConfluenceBreakdown(
    pattern_score=0.7,
    structure_score=0.5,
    level_score=0.6,
    volume_score=0.5,
    regime_score=0.8,
    session_score=0.7,
    risk_score=0.6,
    data_quality_score=0.9,
    weighted_total=0.68,
)


def _make_signal(**overrides) -> SignalOutput:
    """Create a SignalOutput with sensible defaults, overriding any fields."""
    defaults = dict(
        symbol="BTC/USDT",
        asset_class=AssetClass.CRYPTO,
        strategy_name="TestStrategy",
        timeframe=Timeframe.ONE_HOUR,
        action=SignalAction.BUY,
        confidence=0.75,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=99.0,
        entry_zone_high=101.0,
        stop_loss=95.0,
        take_profit_1=110.0,
        take_profit_2=120.0,
        estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP,
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=_CONFLUENCE,
        patterns_detected=["hammer"],
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestBotAgentName:
    def test_bot_name(self):
        bot = ConcreteBotForTest()
        assert bot.name == "TestBot"


class TestBotAgentSignalFiltering:
    def test_bot_takes_signal(self):
        """Confidence 0.75 satisfies >= 0.6 filter."""
        bot = ConcreteBotForTest()
        signal = _make_signal(confidence=0.75)
        assert bot.should_take_signal(signal) is True

    def test_bot_skips_low_confidence(self):
        """Confidence 0.3 does not satisfy >= 0.6 filter — on_signal returns None."""
        bot = ConcreteBotForTest()
        signal = _make_signal(confidence=0.3)
        result = bot.on_signal(signal)
        assert result is None

    def test_bot_skips_no_trade(self):
        """NO_TRADE action → on_signal returns None regardless of confidence."""
        bot = ConcreteBotForTest()
        signal = _make_signal(action=SignalAction.NO_TRADE, confidence=0.9)
        result = bot.on_signal(signal)
        assert result is None


class TestBotAgentTradeOpening:
    def test_bot_opens_trade_on_valid_signal(self):
        """Valid BUY signal → on_signal returns dict with trade_id, balance reduced."""
        bot = ConcreteBotForTest(initial_capital=1_000.0)
        initial_balance = bot.portfolio.balance
        signal = _make_signal(confidence=0.75)

        result = bot.on_signal(signal)

        assert result is not None
        assert "trade_id" in result
        assert result["bot"] == "TestBot"
        assert result["size"] > 0
        assert bot.portfolio.balance < initial_balance


class TestBotAgentExits:
    def test_bot_check_exits_tp1(self):
        """BUY trade with tp1=110, live price 110 → closes with profit."""
        bot = ConcreteBotForTest(initial_capital=10_000.0)
        signal = _make_signal(
            confidence=0.75,
            entry_zone_low=99.0,
            entry_zone_high=101.0,
            stop_loss=95.0,
            take_profit_1=110.0,
        )
        open_result = bot.on_signal(signal)
        assert open_result is not None

        # Live price hits TP1
        live_prices = {"BTC/USDT": 110.0}
        closed = bot.check_exits(live_prices)

        assert len(closed) == 1
        assert closed[0]["realized_pnl"] > 0

    def test_bot_check_exits_sl(self):
        """BUY trade with stop_loss=95, live price 94 → closes with loss."""
        bot = ConcreteBotForTest(initial_capital=10_000.0)
        signal = _make_signal(
            confidence=0.75,
            entry_zone_low=99.0,
            entry_zone_high=101.0,
            stop_loss=95.0,
            take_profit_1=110.0,
        )
        open_result = bot.on_signal(signal)
        assert open_result is not None

        # Live price hits SL
        live_prices = {"BTC/USDT": 94.0}
        closed = bot.check_exits(live_prices)

        assert len(closed) == 1
        assert closed[0]["realized_pnl"] < 0


class TestBotAgentStats:
    def test_bot_stats(self):
        """Stats dict should include required keys and phase='cold_start' on new bot."""
        bot = ConcreteBotForTest()
        stats = bot.get_stats()

        assert stats["phase"] == "cold_start"
        assert "kelly_fraction" in stats
        assert "is_paused" in stats
        assert stats["is_paused"] is False
        assert "consecutive_wins" in stats
        assert "consecutive_losses" in stats

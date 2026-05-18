"""
Unit tests for libs/paper_trading/engine.py — PaperTradingEngine.

Tests cover:
  - Engine creates exactly 6 bots
  - Capital is distributed equally (~10 000 total)
  - dispatch_signal fans signals to all bots and returns list
  - get_summary reports correct totals
  - check_all_exits closes trades that hit TP
  - pause_bot / resume_bot toggle bot state
"""
from __future__ import annotations

import asyncio

import pytest

from libs.core.events.bus import EventBus
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
from libs.paper_trading.engine import INITIAL_CAPITAL, PaperTradingEngine


# ── Signal / Confluence factories ──────────────────────────────────────────────

_CONFLUENCE = ConfluenceBreakdown(
    pattern_score=0.8,
    structure_score=0.7,
    level_score=0.7,
    volume_score=0.8,
    regime_score=0.9,
    session_score=0.8,
    risk_score=0.7,
    data_quality_score=1.0,
    weighted_total=0.80,
)


def _make_signal(**overrides) -> SignalOutput:
    """Create a trending BUY signal that MomentumBot should accept."""
    defaults = dict(
        symbol="BTC/USDT",
        asset_class=AssetClass.CRYPTO,
        strategy_name="ema_crossover",           # accepted by MomentumBot
        timeframe=Timeframe.ONE_HOUR,             # accepted by MomentumBot
        action=SignalAction.BUY,
        confidence=0.85,
        higher_tf_bias=TrendDirection.UPTREND,
        entry_zone_low=99.0,
        entry_zone_high=101.0,
        stop_loss=90.0,
        take_profit_1=115.0,
        take_profit_2=130.0,
        estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP,   # accepted by MomentumBot
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        confluence=_CONFLUENCE,
        patterns_detected=["hammer", "engulfing"],
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestEngineCreation:
    def test_engine_creates_6_bots(self):
        """Engine must create exactly one instance per bot class (6 total)."""
        engine = PaperTradingEngine()
        assert len(engine.bots) == 6

    def test_engine_distributes_capital(self):
        """Sum of all bot balances must equal INITIAL_CAPITAL."""
        engine = PaperTradingEngine()
        total = sum(bot.portfolio.balance for bot in engine.bots)
        assert abs(total - INITIAL_CAPITAL) < 1.0  # within $1 rounding tolerance

    def test_engine_distributes_capital_custom(self):
        """Custom capital is split equally too."""
        engine = PaperTradingEngine(initial_capital=6000.0)
        total = sum(bot.portfolio.balance for bot in engine.bots)
        assert abs(total - 6000.0) < 1.0


class TestDispatchSignal:
    def test_engine_dispatch_signal_returns_list(self):
        """dispatch_signal always returns a list (may be empty if no bot acts)."""
        engine = PaperTradingEngine()
        signal = _make_signal()
        results = engine.dispatch_signal(signal)
        assert isinstance(results, list)

    def test_engine_dispatch_signal_momentum_buy(self):
        """A strong trending BUY should be picked up by at least MomentumBot."""
        engine = PaperTradingEngine()
        signal = _make_signal()
        results = engine.dispatch_signal(signal)
        # At least MomentumBot should open a trade
        assert len(results) >= 1
        # Each result must contain trade_id and bot fields
        for r in results:
            assert "trade_id" in r
            assert "bot" in r

    def test_engine_dispatch_no_trade_signal(self):
        """NO_TRADE signals should be ignored by all bots."""
        engine = PaperTradingEngine()
        signal = _make_signal(action=SignalAction.NO_TRADE)
        results = engine.dispatch_signal(signal)
        assert results == []


class TestEngineSummary:
    def test_engine_summary_structure(self):
        """Summary has all required top-level keys."""
        engine = PaperTradingEngine()
        summary = engine.get_summary({})
        required_keys = {
            "total_balance",
            "initial_capital",
            "total_pnl",
            "total_pnl_pct",
            "total_open_positions",
            "uptime_seconds",
            "started_at",
            "bots",
        }
        assert required_keys.issubset(summary.keys())

    def test_engine_summary_fresh_state(self):
        """Fresh engine: total_balance ≈ 10 000, total_pnl ≈ 0, 6 bots."""
        engine = PaperTradingEngine()
        summary = engine.get_summary({})
        assert abs(summary["total_balance"] - INITIAL_CAPITAL) < 1.0
        assert abs(summary["total_pnl"]) < 1.0
        assert len(summary["bots"]) == 6

    def test_engine_summary_bot_entries_have_extra_fields(self):
        """Each bot entry in summary must include unrealized_pnl and effective_balance."""
        engine = PaperTradingEngine()
        summary = engine.get_summary({})
        for entry in summary["bots"]:
            assert "unrealized_pnl" in entry
            assert "effective_balance" in entry

    def test_engine_summary_bots_sorted_by_effective_balance(self):
        """Bot list in summary is sorted descending by effective_balance."""
        engine = PaperTradingEngine()
        summary = engine.get_summary({})
        balances = [b["effective_balance"] for b in summary["bots"]]
        assert balances == sorted(balances, reverse=True)


class TestCheckExits:
    def test_engine_check_exits_returns_list(self):
        """check_all_exits returns a list even when no trades are open."""
        engine = PaperTradingEngine()
        closed = engine.check_all_exits({"BTC/USDT": 100.0})
        assert isinstance(closed, list)

    def test_engine_check_exits_closes_tp(self):
        """After opening a BUY trade, a price >= TP1 should close it."""
        engine = PaperTradingEngine()
        signal = _make_signal(
            entry_zone_low=99.0,
            entry_zone_high=101.0,
            stop_loss=90.0,
            take_profit_1=115.0,
        )
        # Open the trade
        results = engine.dispatch_signal(signal)
        assert len(results) >= 1, "Pre-condition: at least one bot must open a trade"

        # Simulate price reaching TP
        closed = engine.check_all_exits({"BTC/USDT": 120.0})
        assert len(closed) >= 1
        assert all(c["status"] == "TAKE_PROFIT" for c in closed)


class TestPauseResumeBot:
    def test_engine_pause_bot(self):
        """pause_bot sets is_paused on the named bot and returns True."""
        engine = PaperTradingEngine()
        result = engine.pause_bot("MomentumBot")
        assert result is True

        momentum = next(b for b in engine.bots if b.name == "MomentumBot")
        assert momentum.is_paused is True

    def test_engine_resume_bot(self):
        """resume_bot clears is_paused on a previously paused bot."""
        engine = PaperTradingEngine()
        engine.pause_bot("MomentumBot")
        result = engine.resume_bot("MomentumBot")
        assert result is True

        momentum = next(b for b in engine.bots if b.name == "MomentumBot")
        assert momentum.is_paused is False

    def test_engine_pause_unknown_bot(self):
        """pause_bot returns False when the bot name is not found."""
        engine = PaperTradingEngine()
        assert engine.pause_bot("NonExistentBot") is False

    def test_engine_paused_bot_skips_signal(self):
        """A paused bot should not open new trades."""
        engine = PaperTradingEngine()
        engine.pause_bot("MomentumBot")
        signal = _make_signal()
        results = engine.dispatch_signal(signal)
        # MomentumBot should not be in results
        bot_names = [r["bot"] for r in results]
        assert "MomentumBot" not in bot_names


class TestReset:
    def test_engine_reset_restores_capital(self):
        """After reset, total balance returns to INITIAL_CAPITAL."""
        engine = PaperTradingEngine()
        # Open a trade first
        engine.dispatch_signal(_make_signal())
        engine.reset()
        total = sum(bot.portfolio.balance for bot in engine.bots)
        assert abs(total - INITIAL_CAPITAL) < 1.0

    def test_engine_reset_clears_snapshots(self):
        """reset() clears equity snapshots."""
        engine = PaperTradingEngine()
        engine.snapshot_equity({})
        engine.reset()
        assert engine.get_equity_curve() == []


class TestEquitySnapshot:
    def test_snapshot_records_entry(self):
        """snapshot_equity appends an entry with required keys."""
        engine = PaperTradingEngine()
        engine.snapshot_equity({"BTC/USDT": 100.0})
        curve = engine.get_equity_curve()
        assert len(curve) == 1
        entry = curve[0]
        assert "timestamp" in entry
        assert "total_balance" in entry
        assert "bots" in entry

    def test_snapshot_total_matches_sum(self):
        """Snapshot total_balance should equal sum of per-bot balances."""
        engine = PaperTradingEngine()
        engine.snapshot_equity({})
        entry = engine.get_equity_curve()[0]
        per_bot_sum = sum(entry["bots"].values())
        assert abs(entry["total_balance"] - per_bot_sum) < 0.01


class TestOpenPositions:
    def test_get_all_open_positions_empty_initially(self):
        engine = PaperTradingEngine()
        assert engine.get_all_open_positions() == []

    def test_get_all_open_positions_after_trade(self):
        """After a signal opens a trade, open positions should be non-empty."""
        engine = PaperTradingEngine()
        engine.dispatch_signal(_make_signal())
        positions = engine.get_all_open_positions()
        assert len(positions) >= 1
        assert "bot" in positions[0]
        assert "symbol" in positions[0]


class TestClosedTrades:
    def test_get_closed_trades_empty_initially(self):
        engine = PaperTradingEngine()
        assert engine.get_closed_trades() == []

    def test_get_closed_trades_after_exit(self):
        """Closed trades are returned after TP exit."""
        engine = PaperTradingEngine()
        engine.dispatch_signal(_make_signal(take_profit_1=115.0))
        engine.check_all_exits({"BTC/USDT": 120.0})
        closed = engine.get_closed_trades()
        assert len(closed) >= 1


class TestEventBusIntegration:
    def test_start_subscribes_to_bus(self):
        """start() registers exactly one handler on SIGNAL_GENERATED."""
        engine = PaperTradingEngine()
        bus = EventBus()
        asyncio.get_event_loop().run_until_complete(engine.start(bus))
        assert bus.handler_count(EventBus.SIGNAL_GENERATED) == 1

    def test_on_signal_event_dispatches(self):
        """on_signal_event extracts signal from payload and dispatches it."""
        engine = PaperTradingEngine()
        signal = _make_signal()
        payload = {"signal": signal}
        asyncio.get_event_loop().run_until_complete(engine.on_signal_event(payload))
        # At least MomentumBot should have opened a position
        assert len(engine.get_all_open_positions()) >= 1

    def test_on_signal_event_missing_signal(self):
        """on_signal_event handles missing 'signal' key without raising."""
        engine = PaperTradingEngine()
        asyncio.get_event_loop().run_until_complete(engine.on_signal_event({}))
        assert engine.get_all_open_positions() == []

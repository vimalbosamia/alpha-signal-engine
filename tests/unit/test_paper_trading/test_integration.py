"""Integration test — full signal → bot → trade → exit flow."""
from __future__ import annotations

import pytest
from uuid import uuid4

from libs.core.models.domain import (
    AssetClass, ConfluenceBreakdown, DataQualityStatus, MarketRegime,
    SessionType, SignalAction, SignalOutput, Timeframe, TrendDirection,
)
from libs.paper_trading.engine import PaperTradingEngine


_CONFLUENCE = ConfluenceBreakdown(
    pattern_score=0.7, structure_score=0.5, level_score=0.6,
    volume_score=0.5, regime_score=0.8, session_score=0.7,
    risk_score=0.6, data_quality_score=0.9, weighted_total=0.68,
)


def _sig(**overrides) -> SignalOutput:
    defaults = dict(
        signal_id=uuid4(), symbol="BTCUSDT", asset_class=AssetClass.CRYPTO,
        timeframe=Timeframe.FIFTEEN_MIN, action=SignalAction.BUY, confidence=0.75,
        entry_zone_low=100.0, entry_zone_high=100.0, stop_loss=95.0,
        take_profit_1=110.0, take_profit_2=120.0, estimated_risk_reward=2.5,
        market_regime=MarketRegime.TRENDING_UP, strategy_name="ema_crossover",
        higher_tf_bias=TrendDirection.UPTREND,
        session_status=SessionType.CONTINUOUS,
        data_quality_status=DataQualityStatus.CLEAN,
        patterns_detected=["hammer"], explanation="test",
        confluence=_CONFLUENCE,
    )
    defaults.update(overrides)
    return SignalOutput(**defaults)


def test_full_flow_signal_to_profit():
    """Signal → at least one bot opens trade → price hits TP1 → profit."""
    engine = PaperTradingEngine(initial_capital=10000.0)
    initial_summary = engine.get_summary({})

    sig = _sig()
    trades = engine.dispatch_signal(sig)
    assert len(trades) >= 1, "At least one bot should take a trending BUY"

    positions = engine.get_all_open_positions()
    assert len(positions) >= 1

    exits = engine.check_all_exits({"BTCUSDT": 110.0})
    assert len(exits) >= 1

    summary = engine.get_summary({})
    assert summary["total_pnl"] > 0
    assert summary["total_balance"] > initial_summary["total_balance"]


def test_full_flow_signal_to_loss():
    """Signal → bot opens trade → price hits SL → loss."""
    engine = PaperTradingEngine(initial_capital=10000.0)
    sig = _sig()
    engine.dispatch_signal(sig)

    exits = engine.check_all_exits({"BTCUSDT": 94.0})
    assert len(exits) >= 1

    summary = engine.get_summary({})
    assert summary["total_pnl"] < 0


def test_full_flow_multiple_signals():
    """Multiple signals across styles → multiple bots active."""
    engine = PaperTradingEngine(initial_capital=10000.0)

    engine.dispatch_signal(_sig(
        strategy_name="ema_crossover",
        market_regime=MarketRegime.TRENDING_UP,
    ))

    engine.dispatch_signal(_sig(
        strategy_name="hammer_reversal",
        market_regime=MarketRegime.TRENDING_DOWN,
        patterns_detected=["hammer", "pin_bar"],
        symbol="ETHUSDT",
    ))

    positions = engine.get_all_open_positions()
    assert len(positions) >= 1


def test_engine_reset():
    engine = PaperTradingEngine(initial_capital=10000.0)
    engine.dispatch_signal(_sig())
    engine.check_all_exits({"BTCUSDT": 110.0})

    engine.reset()
    summary = engine.get_summary({})
    assert summary["total_balance"] == pytest.approx(10000.0, abs=1.0)
    assert summary["total_pnl"] == pytest.approx(0.0)
    assert len(engine.get_closed_trades()) == 0

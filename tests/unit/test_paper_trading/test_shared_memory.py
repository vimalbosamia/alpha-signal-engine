"""Tests for SharedLossMemory — cross-bot learning."""
from __future__ import annotations

from libs.paper_trading.shared_memory import SharedLossMemory


def test_no_avoid_initially():
    mem = SharedLossMemory()
    avoid, reason = mem.should_avoid("BTCUSDT", "SELL", "ema_crossover", "trending_up")
    assert avoid is False
    assert reason == ""


def test_avoid_after_3_losses_same_symbol_action():
    mem = SharedLossMemory()
    for _ in range(3):
        mem.record_loss("Bot1", "BTCUSDT", "SELL", "strat_a", "trending_up", [], 2.0, 10.0)
    avoid, reason = mem.should_avoid("BTCUSDT", "SELL", "strat_b", "ranging")
    assert avoid is True
    assert "BTCUSDT" in reason


def test_avoid_after_3_losses_same_strategy_action():
    mem = SharedLossMemory()
    for _ in range(3):
        mem.record_loss("Bot1", "ETHUSDT", "BUY", "hammer_reversal", "trending_down", [], 2.0, 5.0)
    avoid, reason = mem.should_avoid("SOLUSDT", "BUY", "hammer_reversal", "trending_up")
    assert avoid is True
    assert "hammer_reversal" in reason


def test_avoid_after_3_losses_same_regime_action():
    mem = SharedLossMemory()
    for _ in range(3):
        mem.record_loss("Bot2", "ADAUSDT", "SELL", "strat_x", "ranging_low_vol", [], 1.5, 8.0)
    avoid, reason = mem.should_avoid("LINKUSDT", "SELL", "strat_y", "ranging_low_vol")
    assert avoid is True
    assert "ranging_low_vol" in reason


def test_wins_reset_block():
    mem = SharedLossMemory()
    for _ in range(3):
        mem.record_loss("Bot1", "BTCUSDT", "SELL", "strat_a", "trending_up", [], 2.0, 10.0)
    # Now blocked
    avoid, _ = mem.should_avoid("BTCUSDT", "SELL", "strat_a", "trending_up")
    assert avoid is True
    # 2 wins reset it
    mem.record_win("Bot2", "BTCUSDT", "SELL", "strat_a", "trending_up")
    mem.record_win("Bot3", "BTCUSDT", "SELL", "strat_a", "trending_up")
    avoid, _ = mem.should_avoid("BTCUSDT", "SELL", "strat_a", "trending_up")
    assert avoid is False


def test_different_action_not_blocked():
    mem = SharedLossMemory()
    for _ in range(3):
        mem.record_loss("Bot1", "BTCUSDT", "SELL", "strat_a", "trending_up", [], 2.0, 10.0)
    # BUY on same symbol should NOT be blocked
    avoid, _ = mem.should_avoid("BTCUSDT", "BUY", "strat_a", "trending_up")
    assert avoid is False


def test_stats():
    mem = SharedLossMemory()
    mem.record_loss("Bot1", "BTCUSDT", "SELL", "strat_a", "trending_up", [], 2.0, 10.0)
    mem.record_win("Bot2", "ETHUSDT", "BUY", "strat_b", "ranging")
    stats = mem.get_stats()
    assert stats["total_losses"] == 1
    assert stats["total_wins"] == 1

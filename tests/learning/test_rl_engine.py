"""
Unit tests for libs/learning/rl_engine.py.

Tests cover:
  1. Q-table state-action value updates
  2. Epsilon-greedy exploration
  3. Best action selection
  4. Reward computation
  5. Persistence round-trip
"""
from __future__ import annotations

from pathlib import Path

import pytest

from libs.learning.rl_engine import (
    RLEngine,
    encode_action,
    encode_state,
    decode_state,
    decode_action,
)


class TestEncoding:
    """State/action encoding helpers."""

    def test_encode_state(self) -> None:
        key = encode_state("trending_up", "high", "up")
        assert key == "trending_up|high|up"

    def test_decode_state(self) -> None:
        result = decode_state("breakout|low|strong_up")
        assert result["regime"] == "breakout"
        assert result["volatility_bucket"] == "low"
        assert result["trend_direction"] == "strong_up"

    def test_encode_action(self) -> None:
        key = encode_action("momentum", "half")
        assert key == "momentum|half"

    def test_decode_action(self) -> None:
        result = decode_action("reversal|aggressive")
        assert result["strategy"] == "reversal"
        assert result["size_bucket"] == "aggressive"


class TestQUpdate:
    """Q-table updates from trade outcomes."""

    def test_positive_reward_increases_q(self) -> None:
        engine = RLEngine()
        state = encode_state("trending_up")
        action = encode_action("momentum")
        engine.update(state, action, reward=1.0)
        q = engine.get_q_value(state, action)
        assert q > 0

    def test_negative_reward_decreases_q(self) -> None:
        engine = RLEngine()
        state = encode_state("trending_up")
        action = encode_action("momentum")
        engine.update(state, action, reward=-1.0)
        q = engine.get_q_value(state, action)
        assert q < 0

    def test_repeated_positive_updates_increase(self) -> None:
        engine = RLEngine()
        state = encode_state("breakout")
        action = encode_action("breakout")
        prev_q = 0.0
        for _ in range(10):
            new_q = engine.update(state, action, reward=1.0)
            assert new_q >= prev_q
            prev_q = new_q

    def test_total_updates_tracked(self) -> None:
        engine = RLEngine()
        state = encode_state("ranging_low_vol")
        action = encode_action("mean_reversion")
        engine.update(state, action, reward=0.5)
        engine.update(state, action, reward=0.5)
        assert engine.total_updates == 2


class TestBestAction:
    """Best action selection from Q-table."""

    def test_best_action_has_highest_q(self) -> None:
        engine = RLEngine()
        state = encode_state("trending_up")
        a1 = encode_action("momentum", "normal")
        a2 = encode_action("reversal", "normal")

        for _ in range(20):
            engine.update(state, a1, reward=5.0)
            engine.update(state, a2, reward=-2.0)

        best, q = engine.get_best_action(state)
        assert best == a1
        assert q > 0

    def test_unknown_state_returns_empty(self) -> None:
        engine = RLEngine()
        best, q = engine.get_best_action("never_seen|medium|flat")
        assert best == ""
        assert q == 0.0


class TestSelectAction:
    """Epsilon-greedy action selection."""

    def test_returns_dict_with_strategy(self) -> None:
        engine = RLEngine()
        result = engine.select_action("trending_up", "medium", "up")
        assert "strategy" in result
        assert "size_bucket" in result

    def test_pure_exploitation_picks_best(self) -> None:
        engine = RLEngine(epsilon=0.0)
        state = encode_state("breakout", "high", "strong_up")
        good_action = encode_action("breakout", "aggressive")

        for _ in range(50):
            engine.update(state, good_action, reward=10.0)

        # With epsilon=0, always exploit
        results = [engine.select_action("breakout", "high", "strong_up") for _ in range(5)]
        assert all(r["strategy"] == "breakout" and r["size_bucket"] == "aggressive" for r in results)


class TestComputeReward:
    """Reward computation from trade attributes."""

    def test_win_positive(self) -> None:
        engine = RLEngine()
        r = engine.compute_reward(pnl=100.0, won=True)
        assert r > 0

    def test_loss_negative(self) -> None:
        engine = RLEngine()
        r = engine.compute_reward(pnl=-50.0, won=False, disciplined_exit=False)
        assert r < 0

    def test_high_rr_bonus(self) -> None:
        engine = RLEngine()
        low = engine.compute_reward(pnl=100.0, rr=1.0, won=True)
        high = engine.compute_reward(pnl=100.0, rr=3.0, won=True)
        assert high > low


class TestPersistence:
    """Save/load round-trip."""

    def test_save_load_preserves_q_table(self, tmp_path: Path) -> None:
        engine = RLEngine()
        s1 = encode_state("trending_up")
        a1 = encode_action("momentum")
        engine.update(s1, a1, reward=5.0)

        s2 = encode_state("ranging_low_vol")
        a2 = encode_action("reversal")
        engine.update(s2, a2, reward=-2.0)

        path = tmp_path / "rl.json"
        engine.save(path)

        engine2 = RLEngine()
        engine2.load(path)

        assert engine2.get_q_value(s1, a1) == engine.get_q_value(s1, a1)
        assert engine2.get_q_value(s2, a2) == engine.get_q_value(s2, a2)

    def test_load_missing_file_no_crash(self, tmp_path: Path) -> None:
        engine = RLEngine()
        engine.load(tmp_path / "nonexistent.json")
        assert engine.total_updates == 0

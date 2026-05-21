"""
Unit tests for libs/learning/reward_engine.py.

Tests cover:
  1. Reward/penalize score updates
  2. Softmax probability computation
  3. Minimum probability floor enforcement
  4. Auto-registration of unknown strategies
  5. Persistence round-trip
  6. Compute reward components
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from libs.learning.reward_engine import RewardEngine


class TestComputeReward:
    """compute_reward produces correct scalar values."""

    def test_winning_trade_positive(self) -> None:
        engine = RewardEngine(strategies=["momentum"])
        reward = engine.compute_reward(pnl=100.0, rr=2.5)
        assert reward > 0

    def test_losing_trade_negative(self) -> None:
        engine = RewardEngine(strategies=["momentum"])
        reward = engine.compute_reward(pnl=-50.0, rr=0.5, disciplined_exit=False)
        assert reward < 0

    def test_high_rr_bonus(self) -> None:
        engine = RewardEngine(strategies=["momentum"])
        low_rr = engine.compute_reward(pnl=100.0, rr=1.0)
        high_rr = engine.compute_reward(pnl=100.0, rr=3.0)
        assert high_rr > low_rr


class TestRewardPenalize:
    """Reward and penalize update strategy scores."""

    def test_reward_increases_score(self) -> None:
        engine = RewardEngine(strategies=["breakout"])
        engine.reward("breakout", pnl=100.0, rr=2.0)
        assert engine._scores["breakout"] > 0

    def test_penalize_decreases_score(self) -> None:
        engine = RewardEngine(strategies=["breakout"])
        engine.penalize("breakout", pnl=-50.0)
        assert engine._scores["breakout"] < 0

    def test_reward_count_tracked(self) -> None:
        engine = RewardEngine(strategies=["trend"])
        engine.reward("trend", pnl=50.0)
        engine.reward("trend", pnl=30.0)
        assert engine._reward_counts["trend"] == 2

    def test_penalty_count_tracked(self) -> None:
        engine = RewardEngine(strategies=["trend"])
        engine.penalize("trend", pnl=-50.0)
        assert engine._penalty_counts["trend"] == 1


class TestProbabilities:
    """Softmax probabilities with minimum floor."""

    def test_initial_uniform(self) -> None:
        engine = RewardEngine(strategies=["a", "b", "c"], min_probability=0.0)
        probs = engine.get_probabilities()
        # All equal scores → uniform
        for p in probs.values():
            assert abs(p - 1 / 3) < 0.01

    def test_sum_to_one(self) -> None:
        engine = RewardEngine(strategies=["a", "b", "c"])
        engine.reward("a", pnl=100.0, rr=3.0)
        engine.penalize("b", pnl=-80.0)
        probs = engine.get_probabilities()
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_min_probability_floor(self) -> None:
        engine = RewardEngine(strategies=["good", "bad"], min_probability=0.1)
        # Massively reward one, penalize other
        for _ in range(20):
            engine.reward("good", pnl=500.0, rr=5.0)
            engine.penalize("bad", pnl=-500.0)
        probs = engine.get_probabilities()
        assert probs["bad"] >= 0.1

    def test_higher_score_higher_probability(self) -> None:
        engine = RewardEngine(strategies=["a", "b"], min_probability=0.0)
        engine.reward("a", pnl=200.0, rr=3.0)
        probs = engine.get_probabilities()
        assert probs["a"] > probs["b"]


class TestAutoRegister:
    """Unknown strategies are auto-registered."""

    def test_auto_register_on_reward(self) -> None:
        engine = RewardEngine(strategies=["existing"])
        engine.reward("new_strategy", pnl=50.0)
        assert "new_strategy" in engine._scores
        assert engine._scores["new_strategy"] > 0

    def test_auto_register_on_penalize(self) -> None:
        engine = RewardEngine(strategies=["existing"])
        engine.penalize("new_strategy", pnl=-50.0)
        assert "new_strategy" in engine._scores


class TestPersistence:
    """Save/load round-trip."""

    def test_save_load_preserves_scores(self, tmp_path: Path) -> None:
        engine = RewardEngine(strategies=["a", "b"])
        engine.reward("a", pnl=100.0, rr=2.0)
        engine.penalize("b", pnl=-50.0)

        path = tmp_path / "reward.json"
        engine.save(path)

        engine2 = RewardEngine(strategies=["a", "b"])
        engine2.load(path)

        assert engine2._scores["a"] == engine._scores["a"]
        assert engine2._scores["b"] == engine._scores["b"]


class TestValidation:
    """Constructor validation."""

    def test_empty_strategies_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RewardEngine(strategies=[])

    def test_negative_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            RewardEngine(strategies=["a"], temperature=-1.0)

"""
Unit tests for libs/learning/reward_engine.py.

8 tests covering initial probabilities, reward/penalty effects, probability
constraints, statistical selection, compute_reward signal, and persistence.
All tests are fully isolated — no shared state between test cases.
"""
from __future__ import annotations

import json

import pytest

from libs.learning.reward_engine import RewardEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _engine(*strategies: str, min_probability: float = 0.05) -> RewardEngine:
    """Return a fresh RewardEngine with the given strategy names."""
    return RewardEngine(list(strategies), min_probability=min_probability)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestInitialProbabilitiesUniform:
    """Fresh engine with equal scores produces roughly uniform probabilities."""

    def test_initial_probabilities_uniform(self) -> None:
        # Arrange
        engine = _engine("alpha", "beta", "gamma")

        # Act
        probs = engine.get_probabilities()

        # Assert — three strategies, all ≈ 1/3
        assert set(probs.keys()) == {"alpha", "beta", "gamma"}
        expected = pytest.approx(1 / 3, abs=1e-6)
        assert probs["alpha"] == expected
        assert probs["beta"] == expected
        assert probs["gamma"] == expected


class TestRewardIncreasesProbability:
    """Repeatedly rewarding a strategy makes it the highest-probability pick."""

    def test_reward_increases_probability(self) -> None:
        # Arrange
        engine = _engine("momentum", "reversal", "breakout")

        # Act — reward "momentum" 10 times with a winning trade
        for _ in range(10):
            engine.reward("momentum", pnl=50.0, rr=2.5)

        probs = engine.get_probabilities()

        # Assert — "momentum" has the highest probability
        assert probs["momentum"] > probs["reversal"]
        assert probs["momentum"] > probs["breakout"]


class TestPenalizeDecreasesProbability:
    """Repeatedly penalising a strategy makes it the lowest-probability pick."""

    def test_penalize_decreases_probability(self) -> None:
        # Arrange
        engine = _engine("momentum", "reversal", "breakout")

        # Act — penalise "reversal" 10 times
        for _ in range(10):
            engine.penalize("reversal", pnl=-50.0)

        probs = engine.get_probabilities()

        # Assert — "reversal" has the lowest probability
        assert probs["reversal"] < probs["momentum"]
        assert probs["reversal"] < probs["breakout"]


class TestProbabilitiesSumToOne:
    """Probabilities must sum to 1.0 after mixed rewards and penalties."""

    def test_probabilities_sum_to_one(self) -> None:
        # Arrange
        engine = _engine("a", "b", "c", "d")
        engine.reward("a", pnl=100.0, rr=3.0)
        engine.penalize("b", pnl=-80.0, holding_losers=True)
        engine.reward("c", pnl=20.0, rr=1.5)
        engine.penalize("d", pnl=-10.0, overtrading=True)

        # Act
        probs = engine.get_probabilities()

        # Assert
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-9)


class TestMinimumProbabilityFloor:
    """A heavily penalised strategy never drops below min_probability."""

    def test_minimum_probability_floor(self) -> None:
        # Arrange
        min_prob = 0.05
        engine = _engine("good", "bad", min_probability=min_prob)

        # Act — hammer "bad" with max penalties
        for _ in range(50):
            engine.penalize("bad", pnl=-500.0, holding_losers=True, overtrading=True)

        probs = engine.get_probabilities()

        # Assert — "bad" must be at or above the floor
        assert probs["bad"] >= min_prob - 1e-9


class TestShouldTakeStrategy:
    """A rewarded strategy is accepted more often than a penalised one (statistical)."""

    def test_should_take_strategy(self) -> None:
        # Arrange
        engine = _engine("favoured", "disfavoured")
        for _ in range(10):
            engine.reward("favoured", pnl=100.0, rr=3.0)
        for _ in range(10):
            engine.penalize("disfavoured", pnl=-100.0)

        # Act — sample 100 times
        samples = 100
        favoured_accepts = sum(engine.should_take("favoured") for _ in range(samples))
        disfavoured_accepts = sum(
            engine.should_take("disfavoured") for _ in range(samples)
        )

        # Assert — favoured is accepted meaningfully more often
        # (loose threshold to avoid flakiness: favoured >= disfavoured + 5)
        assert favoured_accepts >= disfavoured_accepts + 5


class TestComputeRewardSignal:
    """compute_reward returns positive for a good trade and negative for a bad one."""

    def test_compute_reward_good_trade(self) -> None:
        # Arrange
        engine = _engine("x")

        # Act — profitable, high RR, disciplined, aligned, low drawdown
        reward = engine.compute_reward(
            pnl=200.0,
            rr=3.0,
            disciplined_exit=True,
            regime_aligned=True,
            low_drawdown=True,
        )

        # Assert — all factors positive
        assert reward > 0.0

    def test_compute_reward_bad_trade(self) -> None:
        # Arrange
        engine = _engine("x")

        # Act — loss, low RR, undisciplined, misaligned, high drawdown
        reward = engine.compute_reward(
            pnl=-150.0,
            rr=0.5,
            disciplined_exit=False,
            regime_aligned=False,
            low_drawdown=False,
        )

        # Assert — all factors negative
        assert reward < 0.0


class TestPersistence:
    """save() and load() round-trip preserves scores exactly."""

    def test_persistence(self, tmp_path) -> None:
        # Arrange
        engine = _engine("s1", "s2", "s3")
        engine.reward("s1", pnl=50.0, rr=2.5)
        engine.penalize("s2", pnl=-30.0)
        original_scores = dict(engine._scores)

        save_path = tmp_path / "reward_engine.json"

        # Act — save then load into a fresh engine
        engine.save(save_path)
        loaded_engine = _engine("s1", "s2", "s3")
        loaded_engine.load(save_path)

        # Assert — scores are identical
        for strategy, score in original_scores.items():
            assert loaded_engine._scores[strategy] == pytest.approx(score, abs=1e-12)

        # Assert — JSON file is well-formed and contains expected keys
        payload = json.loads(save_path.read_text())
        assert "scores" in payload
        assert "strategies" in payload
        assert payload["strategies"] == ["s1", "s2", "s3"]

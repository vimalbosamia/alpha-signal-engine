"""
Unit tests for libs/learning/pattern_scorer.py.

5 focused tests covering:
  1. record + retrieve basic statistics
  2. decay reduces effective_weight
  3. confidence_adjustment returns correct sign and magnitude
  4. get_top_patterns sorts correctly
  5. JSON persistence round-trip (save + load)

All tests are fully isolated — no shared state, no I/O side-effects except
the tmp_path fixture provided by pytest.
"""
from __future__ import annotations

import pytest

from libs.learning.pattern_scorer import (
    MAX_CONFIDENCE_ADJ,
    MIN_TRADES_FOR_SCORE,
    PatternScore,
    PatternScoreStore,
    get_pattern_store,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _store_with_trades(
    pattern: str,
    regime: str,
    wins: int,
    losses: int,
    rr: float = 1.0,
) -> PatternScoreStore:
    """Return a fresh store pre-populated with *wins* + *losses* trades."""
    store = PatternScoreStore()
    for _ in range(wins):
        store.record(pattern, regime, won=True, rr=rr)
    for _ in range(losses):
        store.record(pattern, regime, won=False, rr=-rr)
    return store


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRecordAndRetrieveScore:
    """record() accumulates statistics; get_score() returns the correct values."""

    def test_record_and_retrieve_score(self) -> None:
        # Arrange
        store = PatternScoreStore()

        # Act — 2 wins, 1 loss
        store.record("hammer", "trending_up", won=True, rr=2.0)
        store.record("hammer", "trending_up", won=True, rr=1.5)
        store.record("hammer", "trending_up", won=False, rr=-1.0)

        # Assert
        score = store.get_score("hammer", "trending_up")
        assert isinstance(score, PatternScore)
        assert score.total_trades == 3
        assert score.win_rate == pytest.approx(2 / 3, rel=1e-6)
        assert score.avg_rr == pytest.approx((2.0 + 1.5 - 1.0) / 3, rel=1e-6)
        assert score.last_updated != ""

    def test_get_score_returns_empty_for_unknown(self) -> None:
        # Arrange
        store = PatternScoreStore()

        # Act
        score = store.get_score("unknown_pattern", "ranging")

        # Assert — empty score, no crash
        assert score.total_trades == 0
        assert score.win_rate == 0.0
        assert score.avg_rr == 0.0


class TestDecayReducesOldWeight:
    """apply_decay() multiplies effective_weight (and other accumulators) by alpha."""

    def test_decay_reduces_old_weight(self) -> None:
        # Arrange — record 10 wins so effective_weight starts at 10.0
        store = PatternScoreStore(decay_alpha=0.97)
        for _ in range(10):
            store.record("engulfing_bull", "ranging", won=True, rr=1.5)

        score = store.get_score("engulfing_bull", "ranging")
        weight_before = score.effective_weight
        wins_before = score.wins
        rr_before = score.total_rr

        # Act
        store.apply_decay()

        # Assert — same object, mutated in place
        assert score.effective_weight < weight_before
        assert score.effective_weight == pytest.approx(weight_before * 0.97, rel=1e-6)
        assert score.wins == pytest.approx(wins_before * 0.97, rel=1e-6)
        assert score.total_rr == pytest.approx(rr_before * 0.97, rel=1e-6)


class TestConfidenceAdjustment:
    """confidence_adjustment() returns correct sign and respects the ±0.15 cap."""

    def test_high_win_rate_gives_positive_adjustment(self) -> None:
        # Arrange — 80% win rate (8 wins, 2 losses = 10 trades ≥ MIN_TRADES_FOR_SCORE)
        store = _store_with_trades("doji_star", "trending_up", wins=8, losses=2)

        # Act
        adj = store.confidence_adjustment("doji_star", "trending_up")

        # Assert
        assert adj > 0.0
        assert adj <= MAX_CONFIDENCE_ADJ

    def test_low_win_rate_gives_negative_adjustment(self) -> None:
        # Arrange — 30% win rate (3 wins, 7 losses = 10 trades ≥ MIN_TRADES_FOR_SCORE)
        store = _store_with_trades("shooting_star", "ranging", wins=3, losses=7)

        # Act
        adj = store.confidence_adjustment("shooting_star", "ranging")

        # Assert
        assert adj < 0.0
        assert adj >= -MAX_CONFIDENCE_ADJ

    def test_insufficient_trades_returns_zero(self) -> None:
        # Arrange — only 4 trades (below MIN_TRADES_FOR_SCORE = 5)
        store = _store_with_trades("hammer", "trending_down", wins=3, losses=1)

        # Act
        adj = store.confidence_adjustment("hammer", "trending_down")

        # Assert
        assert adj == 0.0

    def test_baseline_win_rate_returns_near_zero(self) -> None:
        # Arrange — exactly BASELINE_WIN_RATE (45%)  → 9 wins, 11 losses
        store = _store_with_trades("inside_bar", "ranging", wins=9, losses=11)

        # Act
        adj = store.confidence_adjustment("inside_bar", "ranging")

        # Assert — should be very close to zero
        assert abs(adj) < 0.01


class TestGetTopPatterns:
    """get_top_patterns() returns patterns sorted by win_rate descending."""

    def test_get_top_patterns_sorted_correctly(self) -> None:
        # Arrange — 3 patterns with distinct win-rates; all ≥ MIN_TRADES_FOR_SCORE
        store = PatternScoreStore()
        _fill_store(store, "pattern_A", "trending_up", wins=8, losses=2)   # 80%
        _fill_store(store, "pattern_B", "trending_up", wins=6, losses=4)   # 60%
        _fill_store(store, "pattern_C", "trending_up", wins=3, losses=7)   # 30%

        # Act
        top2 = store.get_top_patterns("trending_up", n=2)

        # Assert
        assert len(top2) == 2
        assert top2[0].pattern_name == "pattern_A"
        assert top2[1].pattern_name == "pattern_B"
        assert top2[0].win_rate > top2[1].win_rate

    def test_get_top_patterns_excludes_insufficient_trades(self) -> None:
        # Arrange — one pattern has < MIN_TRADES_FOR_SCORE
        store = PatternScoreStore()
        _fill_store(store, "good_pattern", "ranging", wins=7, losses=3)   # 10 trades
        # Only 3 trades — should be excluded
        store.record("sparse_pattern", "ranging", won=True, rr=1.0)
        store.record("sparse_pattern", "ranging", won=True, rr=1.0)
        store.record("sparse_pattern", "ranging", won=True, rr=1.0)

        # Act
        top = store.get_top_patterns("ranging", n=5)

        # Assert — sparse_pattern must not appear
        names = [s.pattern_name for s in top]
        assert "sparse_pattern" not in names
        assert "good_pattern" in names

    def test_get_worst_patterns_sorted_ascending(self) -> None:
        # Arrange
        store = PatternScoreStore()
        _fill_store(store, "pattern_X", "ranging", wins=2, losses=8)   # 20%
        _fill_store(store, "pattern_Y", "ranging", wins=5, losses=5)   # 50%
        _fill_store(store, "pattern_Z", "ranging", wins=7, losses=3)   # 70%

        # Act
        worst = store.get_worst_patterns("ranging", n=2)

        # Assert
        assert len(worst) == 2
        assert worst[0].pattern_name == "pattern_X"
        assert worst[0].win_rate < worst[1].win_rate


class TestPersistenceSaveLoad:
    """save() and load() preserve all PatternScore fields."""

    def test_persistence_save_load(self, tmp_path) -> None:
        # Arrange
        store = PatternScoreStore(decay_alpha=0.95)
        store.record("hammer", "trending_up", won=True, rr=2.0)
        store.record("hammer", "trending_up", won=False, rr=-1.0)
        store.record("doji", "ranging", won=True, rr=1.5)

        save_path = tmp_path / "pattern_scores.json"

        # Act
        store.save(save_path)
        loaded = PatternScoreStore.load(save_path)

        # Assert — structural equality
        assert len(loaded._scores) == 2

        hammer = loaded.get_score("hammer", "trending_up")
        assert hammer.total_trades == 2
        assert hammer.win_rate == pytest.approx(0.5, rel=1e-6)
        assert hammer.avg_rr == pytest.approx((2.0 - 1.0) / 2, rel=1e-6)

        doji = loaded.get_score("doji", "ranging")
        assert doji.total_trades == 1

        # Decay alpha is preserved
        assert loaded._alpha == pytest.approx(0.95, rel=1e-9)

    def test_load_missing_file_returns_empty_store(self, tmp_path) -> None:
        # Arrange
        missing = tmp_path / "nonexistent.json"

        # Act
        store = PatternScoreStore.load(missing)

        # Assert
        assert store.get_all_stats()["total_patterns"] == 0


# ── Internal helper ───────────────────────────────────────────────────────────

def _fill_store(
    store: PatternScoreStore,
    pattern: str,
    regime: str,
    wins: int,
    losses: int,
    rr: float = 1.0,
) -> None:
    """Record *wins* + *losses* outcomes into an existing store."""
    for _ in range(wins):
        store.record(pattern, regime, won=True, rr=rr)
    for _ in range(losses):
        store.record(pattern, regime, won=False, rr=-rr)

"""
Unit tests for libs/learning/pattern_scorer.py.

Tests cover:
  1. Recording wins/losses updates stats correctly
  2. Win rate calculation
  3. Confidence adjustment within bounds
  4. Auto-decay triggers at DECAY_INTERVAL_TRADES
  5. Top/worst pattern retrieval
  6. Persistence round-trip (save/load)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from libs.learning.pattern_scorer import (
    BASELINE_WIN_RATE,
    DECAY_INTERVAL_TRADES,
    MAX_CONFIDENCE_ADJ,
    MIN_TRADES_FOR_SCORE,
    PatternScoreStore,
)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRecord:
    """Recording outcomes updates stats."""

    def test_single_win(self) -> None:
        store = PatternScoreStore()
        store.record("hammer", "trending_up", won=True, rr=2.0)

        score = store.get_score("hammer", "trending_up")
        assert score.wins == 1.0
        assert score.losses == 0.0
        assert score.total_rr == 2.0
        assert score.total_trades == 1

    def test_single_loss(self) -> None:
        store = PatternScoreStore()
        store.record("doji", "ranging", won=False, rr=0.3)

        score = store.get_score("doji", "ranging")
        assert score.wins == 0.0
        assert score.losses == 1.0
        assert score.total_rr == 0.3

    def test_multiple_outcomes(self) -> None:
        store = PatternScoreStore()
        for _ in range(3):
            store.record("engulfing", "breakout", won=True, rr=1.5)
        for _ in range(2):
            store.record("engulfing", "breakout", won=False, rr=0.5)

        score = store.get_score("engulfing", "breakout")
        assert score.total_trades == 5
        assert score.wins == 3.0
        assert score.losses == 2.0


class TestWinRate:
    """Win rate is computed correctly."""

    def test_all_wins(self) -> None:
        store = PatternScoreStore()
        for _ in range(5):
            store.record("hammer", "trending_up", won=True, rr=2.0)
        assert store.get_score("hammer", "trending_up").win_rate == 1.0

    def test_all_losses(self) -> None:
        store = PatternScoreStore()
        for _ in range(5):
            store.record("hammer", "trending_up", won=False, rr=0.3)
        assert store.get_score("hammer", "trending_up").win_rate == 0.0

    def test_empty_returns_zero(self) -> None:
        store = PatternScoreStore()
        assert store.get_score("unknown", "unknown").win_rate == 0.0


class TestConfidenceAdjustment:
    """Confidence adjustment bounded within ±MAX_CONFIDENCE_ADJ."""

    def test_insufficient_trades_returns_zero(self) -> None:
        store = PatternScoreStore()
        for _ in range(MIN_TRADES_FOR_SCORE - 1):
            store.record("hammer", "trending_up", won=True, rr=2.0)
        assert store.confidence_adjustment("hammer", "trending_up") == 0.0

    def test_high_win_rate_positive(self) -> None:
        store = PatternScoreStore()
        for _ in range(10):
            store.record("hammer", "trending_up", won=True, rr=2.0)
        adj = store.confidence_adjustment("hammer", "trending_up")
        assert adj > 0
        assert adj <= MAX_CONFIDENCE_ADJ

    def test_low_win_rate_negative(self) -> None:
        store = PatternScoreStore()
        for _ in range(10):
            store.record("hammer", "trending_up", won=False, rr=0.3)
        adj = store.confidence_adjustment("hammer", "trending_up")
        assert adj < 0
        assert adj >= -MAX_CONFIDENCE_ADJ

    def test_unknown_pattern_returns_zero(self) -> None:
        store = PatternScoreStore()
        assert store.confidence_adjustment("nonexistent", "any") == 0.0


class TestDecay:
    """Auto-decay triggers and reduces statistics."""

    def test_decay_reduces_values(self) -> None:
        store = PatternScoreStore(decay_alpha=0.9)
        for _ in range(10):
            store.record("hammer", "trending_up", won=True, rr=2.0)

        before_wins = store.get_score("hammer", "trending_up").wins
        store.apply_decay()
        after_wins = store.get_score("hammer", "trending_up").wins

        assert after_wins < before_wins
        assert abs(after_wins - before_wins * 0.9) < 0.01

    def test_auto_decay_at_interval(self) -> None:
        store = PatternScoreStore(decay_alpha=0.5)
        # Record exactly DECAY_INTERVAL_TRADES outcomes
        for i in range(DECAY_INTERVAL_TRADES):
            store.record("test", "regime", won=True, rr=1.0)

        # After 50 trades with alpha=0.5, auto-decay fired once
        score = store.get_score("test", "regime")
        # Wins would be 50 * 0.5 = 25 (decayed once at trade 50)
        assert score.wins < DECAY_INTERVAL_TRADES


class TestTopWorstPatterns:
    """Top/worst pattern retrieval."""

    def test_top_patterns(self) -> None:
        store = PatternScoreStore()
        # Good pattern
        for _ in range(10):
            store.record("hammer", "trending_up", won=True, rr=2.0)
        # Bad pattern
        for _ in range(10):
            store.record("doji", "trending_up", won=False, rr=0.3)

        top = store.get_top_patterns("trending_up", n=1)
        assert len(top) == 1
        assert top[0].pattern_name == "hammer"

    def test_worst_patterns(self) -> None:
        store = PatternScoreStore()
        for _ in range(10):
            store.record("hammer", "trending_up", won=True, rr=2.0)
        for _ in range(10):
            store.record("doji", "trending_up", won=False, rr=0.3)

        worst = store.get_worst_patterns("trending_up", n=1)
        assert len(worst) == 1
        assert worst[0].pattern_name == "doji"


class TestPersistence:
    """Save/load round-trip preserves data."""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        store = PatternScoreStore()
        for _ in range(7):
            store.record("hammer", "trending_up", won=True, rr=2.0)
        for _ in range(3):
            store.record("hammer", "trending_up", won=False, rr=0.5)

        path = tmp_path / "scores.json"
        store.save(path)

        loaded = PatternScoreStore.load(path)
        score = loaded.get_score("hammer", "trending_up")
        assert score.total_trades == 10
        assert score.wins == 7.0
        assert score.losses == 3.0

    def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        loaded = PatternScoreStore.load(tmp_path / "nonexistent.json")
        assert loaded._trade_count == 0

"""
Unit tests for libs/learning/vector_memory.py.

Tests cover:
  1. Store and retrieve vectors
  2. Cosine similarity search (find_similar)
  3. Batch similarity search
  4. Win rate estimation from similar conditions
  5. Eviction when exceeding max_entries
  6. Persistence round-trip
  7. Context vector building
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libs.learning.vector_memory import (
    REGIME_ENCODING,
    VECTOR_DIM,
    VectorMemory,
    build_context_vector,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_context(**overrides) -> dict:
    """Build a minimal context dict with sensible defaults."""
    ctx = {
        "rsi": 50.0,
        "macd_histogram": 0.0,
        "atr_pct": 1.0,
        "volume_ratio": 1.0,
        "market_regime": "trending_up",
        "volatility_percentile": 50.0,
        "ema_alignment": 1.0,
        "bb_position": 0.5,
        "trend_strength": 25.0,
        "hour": 12,
        "day_of_week": 3,
        "spread_pct": 0.05,
        "funding_rate": 0.01,
        "open_interest_change": 0.0,
    }
    ctx.update(overrides)
    return ctx


# ── Tests ────────────────────────────────────────────────────────────────────

class TestBuildContextVector:
    """build_context_vector produces correct output."""

    def test_correct_dimension(self) -> None:
        vec = build_context_vector(_make_context())
        assert vec.shape == (VECTOR_DIM,)

    def test_regime_encoding(self) -> None:
        vec = build_context_vector(_make_context(market_regime="breakout"))
        # regime_id is index 4
        assert vec[4] == float(REGIME_ENCODING["breakout"])

    def test_unknown_regime_defaults(self) -> None:
        vec = build_context_vector(_make_context(market_regime="alien_regime"))
        assert vec[4] == float(REGIME_ENCODING["unknown"])

    def test_missing_keys_default_to_zero(self) -> None:
        vec = build_context_vector({})
        assert vec.shape == (VECTOR_DIM,)


class TestStore:
    """Storing and retrieving vector entries."""

    def test_store_increases_size(self) -> None:
        vm = VectorMemory(max_entries=100)
        assert vm.size == 0
        vm.store(_make_context(), trade_id="t1", symbol="BTCUSDT", won=True)
        assert vm.size == 1

    def test_eviction_at_max(self) -> None:
        vm = VectorMemory(max_entries=3)
        for i in range(5):
            vm.store(_make_context(), trade_id=f"t{i}")
        assert vm.size == 3


class TestFindSimilar:
    """Cosine similarity search."""

    def test_identical_context_returns_high_similarity(self) -> None:
        vm = VectorMemory()
        ctx = _make_context()
        vm.store(ctx, trade_id="t1", symbol="BTCUSDT", won=True)

        results = vm.find_similar(ctx, k=1)
        assert len(results) == 1
        assert results[0].similarity > 0.99

    def test_different_context_returns_lower_similarity(self) -> None:
        vm = VectorMemory()
        vm.store(_make_context(rsi=90.0, trend_strength=80.0), trade_id="t1")

        results = vm.find_similar(_make_context(rsi=10.0, trend_strength=5.0), k=1)
        assert len(results) == 1
        assert results[0].similarity < 0.99

    def test_empty_memory_returns_empty(self) -> None:
        vm = VectorMemory()
        results = vm.find_similar(_make_context())
        assert results == []

    def test_filter_by_strategy(self) -> None:
        vm = VectorMemory()
        vm.store(_make_context(), trade_id="t1", strategy="momentum")
        vm.store(_make_context(), trade_id="t2", strategy="reversal")

        results = vm.find_similar(_make_context(), filter_strategy="momentum")
        assert all(r.entry.strategy == "momentum" for r in results)

    def test_min_similarity_threshold(self) -> None:
        vm = VectorMemory()
        vm.store(_make_context(rsi=90.0, trend_strength=90.0), trade_id="t1")

        results = vm.find_similar(
            _make_context(rsi=10.0, trend_strength=5.0),
            min_similarity=0.999,
        )
        # Very different contexts should not pass high threshold
        assert len(results) == 0


class TestBatchSimilar:
    """Vectorized batch similarity search."""

    def test_batch_matches_iterative(self) -> None:
        vm = VectorMemory()
        for i in range(10):
            vm.store(
                _make_context(rsi=30.0 + i * 5),
                trade_id=f"t{i}",
                won=i % 2 == 0,
            )

        query = _make_context(rsi=55.0)
        iterative = vm.find_similar(query, k=3)
        batch = vm.find_similar_batch(query, k=3)

        assert len(iterative) == len(batch)
        for it, bt in zip(iterative, batch):
            assert abs(it.similarity - bt.similarity) < 1e-6


class TestWinRateEstimation:
    """Win rate from similar historical conditions."""

    def test_all_wins_returns_high_rate(self) -> None:
        vm = VectorMemory()
        ctx = _make_context()
        for i in range(10):
            vm.store(ctx, trade_id=f"t{i}", won=True, pnl=50.0)

        result = vm.get_win_rate_for_similar(ctx, k=10, min_similarity=0.5)
        assert result["win_rate"] == 1.0
        assert result["sample_size"] == 10

    def test_all_losses_returns_zero(self) -> None:
        vm = VectorMemory()
        ctx = _make_context()
        for i in range(10):
            vm.store(ctx, trade_id=f"t{i}", won=False, pnl=-30.0)

        result = vm.get_win_rate_for_similar(ctx, k=10, min_similarity=0.5)
        assert result["win_rate"] == 0.0

    def test_empty_memory_returns_defaults(self) -> None:
        vm = VectorMemory()
        result = vm.get_win_rate_for_similar(_make_context())
        assert result["win_rate"] == 0.5
        assert result["sample_size"] == 0


class TestPersistence:
    """Save/load round-trip."""

    def test_save_load_preserves_entries(self, tmp_path: Path) -> None:
        vm = VectorMemory()
        ctx = _make_context()
        vm.store(ctx, trade_id="t1", symbol="BTCUSDT", won=True, pnl=100.0)
        vm.store(ctx, trade_id="t2", symbol="ETHUSDT", won=False, pnl=-50.0)

        path = tmp_path / "vectors.json"
        vm.save(path)

        vm2 = VectorMemory()
        vm2.load(path)

        assert vm2.size == 2
        stats = vm2.get_stats()
        assert stats["unique_symbols"] == 2

    def test_load_missing_file_stays_empty(self, tmp_path: Path) -> None:
        vm = VectorMemory()
        vm.load(tmp_path / "nonexistent.json")
        assert vm.size == 0

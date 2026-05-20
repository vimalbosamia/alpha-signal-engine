"""
Unit tests for libs/learning/model_registry.py.

6 tests covering registration, activation, comparison logic, and rollback.
All tests are fully isolated — no I/O, no side-effects.
"""
from __future__ import annotations

import pytest

from libs.learning.model_registry import ModelVersion, ModelVersionRegistry


# ── Helpers ───────────────────────────────────────────────────────────────────

_STRATEGY = "breakout_long"
_WEIGHTS = {"rsi": 0.4, "macd": 0.3, "volume": 0.3}


def _registry_with_active(
    strategy: str = _STRATEGY,
    win_rate: float = 0.60,
    sharpe: float = 1.5,
    trade_count: int = 50,
) -> tuple[ModelVersionRegistry, ModelVersion]:
    """Return a registry with one registered AND activated version."""
    reg = ModelVersionRegistry()
    v = reg.register(strategy, _WEIGHTS, win_rate, sharpe, trade_count)
    reg.activate(strategy, v.version_id)
    return reg, reg.get_active(strategy)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestRegisterVersion:
    """register() creates a version that is NOT active by default."""

    def test_register_version(self) -> None:
        # Arrange
        reg = ModelVersionRegistry()

        # Act
        v = reg.register(_STRATEGY, _WEIGHTS, win_rate=0.65, sharpe_ratio=1.8, trade_count=30)

        # Assert
        assert isinstance(v, ModelVersion)
        assert v.strategy_name == _STRATEGY
        assert v.win_rate == pytest.approx(0.65)
        assert v.sharpe_ratio == pytest.approx(1.8)
        assert v.trade_count == 30
        assert v.is_active is False
        assert v.version_id != ""
        assert v.created_at != ""


class TestActivate:
    """activate() marks the target version as active."""

    def test_activate(self) -> None:
        # Arrange
        reg = ModelVersionRegistry()
        v = reg.register(_STRATEGY, _WEIGHTS, win_rate=0.55, sharpe_ratio=1.2, trade_count=25)
        assert v.is_active is False

        # Act
        result = reg.activate(_STRATEGY, v.version_id)

        # Assert
        assert result is True
        active = reg.get_active(_STRATEGY)
        assert active is not None
        assert active.version_id == v.version_id
        assert active.is_active is True


class TestCompareBetterModel:
    """compare() returns should_switch=True when new model is at least as good."""

    def test_compare_better_model(self) -> None:
        # Arrange — register and activate a baseline version
        reg, active = _registry_with_active(win_rate=0.60, sharpe=1.5, trade_count=50)
        # Register a clearly better new version (same strategy)
        reg.register(
            _STRATEGY, _WEIGHTS,
            win_rate=0.68, sharpe_ratio=1.9, trade_count=30
        )

        # Act
        result = reg.compare(_STRATEGY)

        # Assert
        assert result["should_switch"] is True
        assert result["active"] is not None
        assert result["latest"] is not None
        assert result["latest"].win_rate == pytest.approx(0.68)


class TestCompareWorseModel:
    """compare() returns should_switch=False when win_rate regression is too large."""

    def test_compare_worse_model(self) -> None:
        # Arrange — baseline active with 0.65 win_rate
        reg, active = _registry_with_active(win_rate=0.65, sharpe=1.5, trade_count=50)
        # Register a new version with win_rate well below (active - 0.05 threshold)
        reg.register(
            _STRATEGY, _WEIGHTS,
            win_rate=0.40, sharpe_ratio=1.5, trade_count=30
        )

        # Act
        result = reg.compare(_STRATEGY)

        # Assert
        assert result["should_switch"] is False
        assert "regression" in result["reason"].lower() or "win_rate" in result["reason"]


class TestCompareInsufficientTrades:
    """compare() returns should_switch=False when latest version has < 20 trades."""

    def test_compare_insufficient_trades(self) -> None:
        # Arrange
        reg, _ = _registry_with_active(win_rate=0.60, sharpe=1.5, trade_count=50)
        # Register a new version with only 10 trades
        reg.register(
            _STRATEGY, _WEIGHTS,
            win_rate=0.80, sharpe_ratio=2.5, trade_count=10
        )

        # Act
        result = reg.compare(_STRATEGY)

        # Assert
        assert result["should_switch"] is False
        assert "10" in result["reason"] or "trades" in result["reason"].lower()


class TestRollback:
    """rollback() reverts to the previously active version."""

    def test_rollback(self) -> None:
        # Arrange — register two versions; activate v1, then v2
        reg = ModelVersionRegistry()
        v1 = reg.register(_STRATEGY, _WEIGHTS, win_rate=0.60, sharpe_ratio=1.5, trade_count=50)
        reg.activate(_STRATEGY, v1.version_id)
        v2 = reg.register(_STRATEGY, _WEIGHTS, win_rate=0.65, sharpe_ratio=1.7, trade_count=30)
        reg.activate(_STRATEGY, v2.version_id)

        assert reg.get_active(_STRATEGY).version_id == v2.version_id

        # Act
        success = reg.rollback(_STRATEGY)

        # Assert
        assert success is True
        restored = reg.get_active(_STRATEGY)
        assert restored is not None
        assert restored.version_id == v1.version_id

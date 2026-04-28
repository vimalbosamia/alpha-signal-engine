"""
Unit tests for the three SELL-side strategies.

Coverage:
  - ShootingStarReversalStrategy (4 cases)
  - SupportBreakdownStrategy (3 cases)
  - PullbackBearContinuationStrategy (5 cases)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from libs.core.models.domain import (
    AssetClass,
    SignalAction,
    TrendDirection,
)
from libs.strategies.reversal.shooting_star_reversal import ShootingStarReversalStrategy
from libs.strategies.breakout.support_breakdown import SupportBreakdownStrategy
from libs.strategies.continuation.pullback_bear import PullbackBearContinuationStrategy


# ── DataFrame factory helpers ─────────────────────────────────────────────────


def _make_df(n: int = 60, close: float = 100.0, has_vwap: bool = True) -> pd.DataFrame:
    np.random.seed(42)
    closes = close + np.cumsum(np.random.randn(n) * 0.1)
    df = pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.002,
            "low": closes * 0.997,
            "close": closes,
            "volume": np.full(n, 1000.0),
        }
    )
    if has_vwap:
        df["vwap"] = closes  # VWAP == close for simplicity
    return df


def _make_structure(trend: TrendDirection) -> MagicMock:
    structure = MagicMock()
    structure.trend = trend
    return structure


def _make_resistance_level(price: float) -> MagicMock:
    level = MagicMock()
    level.price = price
    level.level_type = "resistance"
    return level


def _make_support_level(price: float) -> MagicMock:
    level = MagicMock()
    level.price = price
    level.level_type = "support"
    return level


def _call_kwargs(
    strategy,
    df: pd.DataFrame,
    structure=None,
    levels=None,
    volume=None,
    asset_class: AssetClass = AssetClass.STOCK,
):
    """Invoke generate_candidate with sensible defaults."""
    return strategy.generate_candidate(
        symbol="TEST",
        asset_class=asset_class,
        df=df,
        df_htf=None,
        session=None,
        quality=None,
        structure=structure,
        levels=levels or [],
        volume=volume,
        regime=None,
    )


# ── ShootingStarReversalStrategy ──────────────────────────────────────────────


class TestShootingStarReversalStrategy:
    strategy = ShootingStarReversalStrategy()

    def test_shooting_star_no_pattern_returns_none(self):
        """Plain candles with no shooting star should yield None."""
        df = _make_df(n=60, close=100.0)
        # Default synthetic bars have no shooting star geometry
        resistance = _make_resistance_level(100.3)
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.RANGING),
            levels=[resistance],
        )
        assert result is None

    def test_shooting_star_uptrend_blocked(self):
        """UPTREND context should block signal generation."""
        df = _make_df(n=60, close=100.0)
        # Craft a valid shooting star bar
        df.iloc[-1, df.columns.get_loc("open")] = 100.0
        df.iloc[-1, df.columns.get_loc("close")] = 99.5
        df.iloc[-1, df.columns.get_loc("high")] = 105.0
        df.iloc[-1, df.columns.get_loc("low")] = 99.0
        resistance = _make_resistance_level(100.3)
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.UPTREND),
            levels=[resistance],
        )
        assert result is None

    def test_shooting_star_no_resistance_returns_none(self):
        """No matching resistance levels should yield None."""
        df = _make_df(n=60, close=100.0)
        # Craft a valid shooting star bar
        df.iloc[-1, df.columns.get_loc("open")] = 100.0
        df.iloc[-1, df.columns.get_loc("close")] = 99.5
        df.iloc[-1, df.columns.get_loc("high")] = 105.0
        df.iloc[-1, df.columns.get_loc("low")] = 99.0
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.RANGING),
            levels=[],
        )
        assert result is None

    def test_shooting_star_sell_signal(self):
        """Valid shooting star near resistance should produce a SELL candidate."""
        df = _make_df(n=60, close=100.0)
        # Craft last bar with long upper wick: high=105, open=100, close=99.5, low=99
        df.iloc[-1, df.columns.get_loc("open")] = 100.0
        df.iloc[-1, df.columns.get_loc("close")] = 99.5
        df.iloc[-1, df.columns.get_loc("high")] = 105.0
        df.iloc[-1, df.columns.get_loc("low")] = 99.0

        # Resistance must be within MIN_LEVEL_PROXIMITY_PCT (0.5%) of last_close=99.5
        # i.e. within [99.00, 99.99]; use 99.7 (0.2 away)
        resistance = _make_resistance_level(99.7)
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.RANGING),
            levels=[resistance],
        )
        assert result is not None
        assert result.proposed_action == SignalAction.SELL
        entry_mid = (result.entry_zone_low + result.entry_zone_high) / 2.0
        assert result.stop_loss > entry_mid, "stop must be above entry for SELL"
        assert result.take_profit_1 < entry_mid, "take profit must be below entry for SELL"
        assert result.take_profit_1 > 0.0


# ── SupportBreakdownStrategy ──────────────────────────────────────────────────


class TestSupportBreakdownStrategy:
    strategy = SupportBreakdownStrategy()

    def test_breakdown_no_level_returns_none(self):
        """No support levels should yield None."""
        df = _make_df(n=60, close=97.8)
        result = _call_kwargs(self.strategy, df, levels=[])
        assert result is None

    def test_breakdown_insufficient_volume_returns_none(self):
        """Low relative volume (1.0 < 1.3) should block the signal."""
        df = _make_df(n=60, close=97.8)
        support = _make_support_level(98.2)

        volume = MagicMock()
        volume.relative_volume = 1.0  # below MIN_REL_VOL of 1.3

        result = _call_kwargs(
            self.strategy,
            df,
            levels=[support],
            volume=volume,
        )
        assert result is None

    def test_breakdown_sell_signal(self):
        """Close below support with sufficient volume should produce a SELL candidate."""
        df = _make_df(n=60, close=97.8)
        # Support at 98.2 — price broke below it (97.8 < 98.2 * 0.998 = 98.004)
        support = _make_support_level(98.2)

        volume = MagicMock()
        volume.relative_volume = 1.5  # above MIN_REL_VOL

        result = _call_kwargs(
            self.strategy,
            df,
            levels=[support],
            volume=volume,
        )
        assert result is not None
        assert result.proposed_action == SignalAction.SELL
        entry_mid = (result.entry_zone_low + result.entry_zone_high) / 2.0
        assert result.stop_loss > entry_mid, "stop must be above entry for SELL"
        assert result.take_profit_1 < entry_mid, "take profit must be below entry for SELL"
        assert result.take_profit_1 > 0.0


# ── PullbackBearContinuationStrategy ─────────────────────────────────────────


class TestPullbackBearContinuationStrategy:
    strategy = PullbackBearContinuationStrategy()

    def test_pullback_bear_uptrend_returns_none(self):
        """UPTREND context should block signal generation."""
        df = _make_df(n=60, close=100.0)
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.UPTREND),
        )
        assert result is None

    def test_pullback_bear_no_vwap_column_returns_none(self):
        """DataFrame without a 'vwap' column should yield None."""
        df = _make_df(n=60, close=100.0, has_vwap=False)
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.DOWNTREND),
        )
        assert result is None

    def test_pullback_bear_not_near_vwap_returns_none(self):
        """Close far from VWAP (> 0.5%) should yield None."""
        df = _make_df(n=60, close=100.0)
        # Set VWAP far above close so proximity > PULLBACK_VWAP_PCT
        df["vwap"] = 102.0  # 2% above close
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.DOWNTREND),
        )
        assert result is None

    def test_pullback_bear_bullish_bar_returns_none(self):
        """Last bar with close > open (bullish) should yield None."""
        df = _make_df(n=60, close=100.0)
        # VWAP == close so proximity is 0 — but bar is bullish
        df["vwap"] = df["close"]
        df.iloc[-1, df.columns.get_loc("open")] = 99.0    # open < close → bullish
        df.iloc[-1, df.columns.get_loc("close")] = 100.0
        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.DOWNTREND),
        )
        assert result is None

    def test_pullback_bear_sell_signal(self):
        """Valid downtrend + bearish bar at VWAP should produce a SELL candidate."""
        df = _make_df(n=60, close=100.0)
        # Set last bar to bearish values first, then set VWAP = close for that bar
        df.iloc[-1, df.columns.get_loc("open")] = 100.5   # open > close → bearish
        df.iloc[-1, df.columns.get_loc("close")] = 100.0
        # Assign vwap after modifying close so the last row's vwap == last row's close
        df["vwap"] = df["close"]

        result = _call_kwargs(
            self.strategy,
            df,
            structure=_make_structure(TrendDirection.DOWNTREND),
        )
        assert result is not None
        assert result.proposed_action == SignalAction.SELL
        entry_mid = (result.entry_zone_low + result.entry_zone_high) / 2.0
        assert result.stop_loss > entry_mid, "stop must be above entry for SELL"
        assert result.take_profit_1 < entry_mid, "take profit must be below entry for SELL"
        assert result.take_profit_1 > 0.0

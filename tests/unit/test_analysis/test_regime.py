"""
Unit tests for RegimeEngine.

All DataFrames are built synthetically — no external data or fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from libs.analysis.regime.engine import RegimeAnalysis, RegimeEngine
from libs.core.models.domain import MarketRegime


# ── Helper ─────────────────────────────────────────────────────────────────────

def make_df(closes, highs=None, lows=None):
    n = len(closes)
    closes = np.array(closes, dtype=float)
    highs = np.array(highs, dtype=float) if highs is not None else closes * 1.005
    lows = np.array(lows, dtype=float) if lows is not None else closes * 0.995
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
    )


# ── Fixtures / shared data ─────────────────────────────────────────────────────

@pytest.fixture
def engine():
    return RegimeEngine()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAtrCalculation:
    """ATR must be positive and proportional to bar range."""

    def test_atr_positive_on_valid_input(self, engine):
        """ATR > 0 for a DataFrame with at least 15 bars."""
        closes = [100.0 + i * 0.1 for i in range(20)]
        df = make_df(closes)
        result = engine.analyze(df)
        assert result.atr > 0.0

    def test_atr_pct_formula(self, engine):
        """atr_pct should equal atr / last_close * 100, approximately."""
        closes = [200.0] * 20
        df = make_df(closes)
        result = engine.analyze(df)
        expected = result.atr / float(df["close"].iloc[-1]) * 100.0
        assert abs(result.atr_pct - expected) < 1e-9


class TestRegimeClassification:
    """Regime labels match market conditions."""

    def test_trending_up_on_rising_closes(self, engine):
        """60 bars of steadily rising closes → TRENDING_UP or BREAKOUT."""
        closes = [100.0 + i * 0.5 for i in range(60)]
        df = make_df(closes)
        result = engine.analyze(df)
        assert result.regime in (MarketRegime.TRENDING_UP, MarketRegime.BREAKOUT)

    def test_trending_down_on_falling_closes(self, engine):
        """60 bars of steadily falling closes → TRENDING_DOWN or BREAKOUT."""
        closes = [200.0 - i * 0.5 for i in range(60)]
        df = make_df(closes)
        result = engine.analyze(df)
        assert result.regime in (MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT)

    def test_ranging_low_vol_on_zigzag(self, engine):
        """60 bars of tiny zigzag closes → RANGING_LOW_VOL."""
        base = 100.0
        # very small alternating moves — minimal ATR, no trend
        closes = [base + (0.01 if i % 2 == 0 else -0.01) for i in range(60)]
        highs = [c + 0.005 for c in closes]
        lows = [c - 0.005 for c in closes]
        df = make_df(closes, highs=highs, lows=lows)
        result = engine.analyze(df)
        assert result.regime == MarketRegime.RANGING_LOW_VOL

    def test_high_vol_spike_regime(self, engine):
        """Last bar with 10× the normal range → RANGING_HIGH_VOL or CLIMACTIC."""
        closes = [100.0] * 60
        highs = [100.5] * 60
        lows = [99.5] * 60
        # Make the last bar extremely wide
        highs[-1] = 105.0
        lows[-1] = 95.0
        df = make_df(closes, highs=highs, lows=lows)
        result = engine.analyze(df)
        assert result.regime in (MarketRegime.RANGING_HIGH_VOL, MarketRegime.CLIMACTIC)


class TestExpansionContraction:
    """is_expanding / is_contracting flags."""

    def test_is_expanding_when_atr_grows(self):
        """Force a regime where ATR grows sharply — is_expanding should be True."""
        # Stable bars then a very large final bar → ATR jumps
        closes = [100.0] * 30
        highs = [100.2] * 30
        lows = [99.8] * 30
        # Last bar: huge range to spike ATR
        highs[-1] = 110.0
        lows[-1] = 90.0
        df = make_df(closes, highs=highs, lows=lows)
        engine = RegimeEngine()
        result = engine.analyze(df)
        assert result.is_expanding is True

    def test_is_contracting_when_atr_shrinks(self):
        """Start with volatile bars, end with tiny bars — is_contracting should be True."""
        n = 40
        closes = [100.0] * n
        # Wide bars for first 38 bars, tiny bars for last 2
        highs = [103.0] * n
        lows = [97.0] * n
        highs[-1] = 100.01
        lows[-1] = 99.99
        highs[-2] = 100.01
        lows[-2] = 99.99
        df = make_df(closes, highs=highs, lows=lows)
        engine = RegimeEngine()
        result = engine.analyze(df)
        # ATR should have dropped from prior bar to last bar
        assert result.is_contracting is True


class TestEdgeCases:
    """Engine must never raise and must return safe defaults on bad input."""

    def test_empty_df_returns_unknown(self, engine):
        """Empty DataFrame → UNKNOWN, no exception."""
        df = pd.DataFrame()
        result = engine.analyze(df)
        assert result.regime == MarketRegime.UNKNOWN
        assert result.atr == 0.0
        assert result.atr_pct == 0.0
        assert result.vol_score == 0.5

    def test_two_bar_df_returns_unknown(self, engine):
        """DataFrame with only 2 bars → UNKNOWN, no exception."""
        df = make_df([100.0, 101.0])
        result = engine.analyze(df)
        assert result.regime == MarketRegime.UNKNOWN

    def test_none_df_returns_unknown(self, engine):
        """None input → UNKNOWN, no exception."""
        result = engine.analyze(None)
        assert result.regime == MarketRegime.UNKNOWN

    def test_missing_columns_returns_unknown(self, engine):
        """DataFrame missing required columns → UNKNOWN, no exception."""
        df = pd.DataFrame({"close": [100.0] * 20})
        result = engine.analyze(df)
        assert result.regime == MarketRegime.UNKNOWN


class TestVolScore:
    """vol_score must always be in [0.0, 1.0] regardless of regime."""

    def test_vol_score_bounded_for_all_regimes(self):
        """vol_score is always in [0.0, 1.0] for any market condition."""
        engine = RegimeEngine()

        scenarios = [
            # Trending up
            [100.0 + i * 0.5 for i in range(60)],
            # Trending down
            [200.0 - i * 0.5 for i in range(60)],
            # Ranging
            [100.0 + (0.01 if i % 2 == 0 else -0.01) for i in range(60)],
            # Flat
            [100.0] * 60,
        ]

        for closes in scenarios:
            df = make_df(closes)
            result = engine.analyze(df)
            assert 0.0 <= result.vol_score <= 1.0, (
                f"vol_score {result.vol_score} out of range for regime {result.regime}"
            )

    def test_vol_score_bounded_on_climactic_spike(self):
        """vol_score stays in [0.0, 1.0] even for an extreme climactic spike."""
        closes = [100.0] * 60
        highs = [100.5] * 60
        lows = [99.5] * 60
        highs[-1] = 130.0
        lows[-1] = 70.0
        df = make_df(closes, highs=highs, lows=lows)
        engine = RegimeEngine()
        result = engine.analyze(df)
        assert 0.0 <= result.vol_score <= 1.0

    def test_low_vol_range_score_above_midpoint(self):
        """Low-volatility ranging markets should have a vol_score above 0.5."""
        closes = [100.0 + (0.01 if i % 2 == 0 else -0.01) for i in range(60)]
        highs = [c + 0.005 for c in closes]
        lows = [c - 0.005 for c in closes]
        df = make_df(closes, highs=highs, lows=lows)
        engine = RegimeEngine()
        result = engine.analyze(df)
        if result.regime == MarketRegime.RANGING_LOW_VOL:
            assert result.vol_score > 0.5


class TestRegimeAnalysisDataclass:
    """RegimeAnalysis is a frozen dataclass with correct defaults."""

    def test_result_is_frozen(self, engine):
        """RegimeAnalysis should be immutable (frozen dataclass)."""
        closes = [100.0 + i * 0.1 for i in range(20)]
        df = make_df(closes)
        result = engine.analyze(df)
        with pytest.raises((AttributeError, TypeError)):
            result.regime = MarketRegime.UNKNOWN  # type: ignore[misc]

    def test_ema_fast_and_slow_populated(self, engine):
        """ema_fast and ema_slow should be floats on valid input."""
        closes = [100.0 + i * 0.1 for i in range(30)]
        df = make_df(closes)
        result = engine.analyze(df)
        assert isinstance(result.ema_fast, float)
        assert isinstance(result.ema_slow, float)

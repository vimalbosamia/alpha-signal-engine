"""
Unit tests for IndicatorsEngine.

All DataFrames are built synthetically — no external data or fixtures.
Tests follow TDD: written before implementation, each test targets a
specific contract of IndicatorSnapshot / IndicatorsEngine.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from libs.analysis.indicators.engine import IndicatorSnapshot, IndicatorsEngine


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_ohlcv(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
    include_vwap: bool = False,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame with optional vwap column."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    highs_arr = np.array(highs, dtype=float) if highs is not None else closes_arr * 1.005
    lows_arr = np.array(lows, dtype=float) if lows is not None else closes_arr * 0.995
    vols_arr = np.array(volumes, dtype=float) if volumes is not None else np.full(n, 1_000_000.0)

    df = pd.DataFrame(
        {
            "open": closes_arr * 0.999,
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
            "volume": vols_arr,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
    )
    if include_vwap:
        # simple proxy: typical price
        df["vwap"] = (highs_arr + lows_arr + closes_arr) / 3.0
    return df


def make_trending_up(n: int = 250, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """Steadily rising price series."""
    closes = [start + i * step for i in range(n)]
    return make_ohlcv(closes)


def make_flat(n: int = 250, price: float = 100.0) -> pd.DataFrame:
    """Flat price series with small noise."""
    rng = np.random.default_rng(42)
    closes = price + rng.uniform(-0.1, 0.1, n)
    return make_ohlcv(closes.tolist())


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine() -> IndicatorsEngine:
    return IndicatorsEngine()


@pytest.fixture
def df_250() -> pd.DataFrame:
    """250-bar trending up DataFrame — sufficient for all indicators."""
    return make_trending_up(250)


@pytest.fixture
def df_10() -> pd.DataFrame:
    """10-bar DataFrame — insufficient for most indicators."""
    return make_trending_up(10)


# ── Contract: sufficient data → all indicators populated ─────────────────────

class TestSufficientData:
    def test_compute_returns_indicator_snapshot(self, engine, df_250):
        snap = engine.compute(df_250)
        assert isinstance(snap, IndicatorSnapshot)

    def test_ema_9_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.ema_9 is not None

    def test_ema_20_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.ema_20 is not None

    def test_ema_50_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.ema_50 is not None

    def test_ema_200_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.ema_200 is not None

    def test_sma_values_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.sma_20 is not None
        assert snap.sma_50 is not None
        assert snap.sma_200 is not None

    def test_rsi_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.rsi is not None

    def test_rsi_prev_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.rsi_prev is not None

    def test_macd_fields_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.macd_line is not None
        assert snap.macd_signal is not None
        assert snap.macd_histogram is not None
        assert snap.macd_histogram_prev is not None

    def test_bollinger_fields_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.bb_upper is not None
        assert snap.bb_middle is not None
        assert snap.bb_lower is not None
        assert snap.bb_width is not None
        assert snap.bb_pct_b is not None

    def test_atr_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.atr is not None

    def test_adx_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.adx is not None

    def test_stoch_rsi_fields_not_none(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.stoch_rsi_k is not None
        assert snap.stoch_rsi_d is not None

    def test_ema_9_positive(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.ema_9 > 0.0

    def test_atr_positive(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.atr > 0.0


# ── Contract: insufficient data → long-period indicators are None ─────────────

class TestInsufficientData:
    def test_ema_200_is_none_with_10_bars(self, engine, df_10):
        snap = engine.compute(df_10)
        assert snap.ema_200 is None

    def test_rsi_is_none_with_10_bars(self, engine, df_10):
        snap = engine.compute(df_10)
        assert snap.rsi is None

    def test_macd_is_none_with_10_bars(self, engine, df_10):
        snap = engine.compute(df_10)
        assert snap.macd_line is None

    def test_adx_is_none_with_insufficient_bars(self, engine):
        df = make_trending_up(5)
        snap = engine.compute(df)
        assert snap.adx is None


# ── Contract: never raises ────────────────────────────────────────────────────

class TestNeverRaises:
    def test_empty_dataframe_returns_snapshot(self, engine):
        snap = engine.compute(pd.DataFrame())
        assert isinstance(snap, IndicatorSnapshot)

    def test_empty_dataframe_all_none(self, engine):
        snap = engine.compute(pd.DataFrame())
        # Every field should be None when there is no data
        for field_name in IndicatorSnapshot.__dataclass_fields__:
            assert getattr(snap, field_name) is None, f"{field_name} should be None"

    def test_single_row_returns_snapshot(self, engine):
        df = make_ohlcv([100.0])
        snap = engine.compute(df)
        assert isinstance(snap, IndicatorSnapshot)

    def test_dataframe_with_nan_close_does_not_raise(self, engine):
        df = make_trending_up(30)
        df.loc[df.index[-1], "close"] = float("nan")
        snap = engine.compute(df)
        assert isinstance(snap, IndicatorSnapshot)


# ── Contract: value correctness ───────────────────────────────────────────────

class TestEMAOrdering:
    def test_ema_ordering_in_clear_uptrend(self, engine):
        """In a clear uptrend: ema_9 > ema_20 > ema_50."""
        df = make_trending_up(250, step=1.0)
        snap = engine.compute(df)
        assert snap.ema_9 is not None
        assert snap.ema_20 is not None
        assert snap.ema_50 is not None
        assert snap.ema_9 > snap.ema_20 > snap.ema_50


class TestRSIRange:
    def test_rsi_between_0_and_100(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.rsi is not None
        assert 0.0 <= snap.rsi <= 100.0

    def test_rsi_prev_between_0_and_100(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.rsi_prev is not None
        assert 0.0 <= snap.rsi_prev <= 100.0

    def test_rsi_high_in_strong_uptrend(self, engine):
        """Strong uptrend → RSI should be elevated (>50)."""
        df = make_trending_up(250, step=2.0)
        snap = engine.compute(df)
        assert snap.rsi is not None
        assert snap.rsi > 50.0


class TestBollingerBands:
    def test_bb_ordering(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.bb_upper > snap.bb_middle > snap.bb_lower

    def test_bb_width_formula(self, engine, df_250):
        """bb_width = (upper - lower) / middle."""
        snap = engine.compute(df_250)
        expected = (snap.bb_upper - snap.bb_lower) / snap.bb_middle
        assert abs(snap.bb_width - expected) < 1e-6

    def test_bb_pct_b_above_1_when_price_above_upper(self, engine):
        """When the last close is above the upper band, pct_b should exceed 1.0.

        We build a minimal flat series so the BB range is tight, then set the
        last close to a value well above the BB upper band.
        """
        # Flat price of 100 → BB bands will be very narrow (small std dev)
        n = 50
        closes = [100.0] * n
        df = make_ohlcv(closes)
        # Force last row's close to be well above 100 + any plausible upper band
        # Typical std ≈ 0 here, so upper band ≈ 100; set close >> 100
        df.iloc[-1, df.columns.get_loc("close")] = 110.0
        df.iloc[-1, df.columns.get_loc("high")] = 111.0
        snap = engine.compute(df)
        assert snap.bb_pct_b is not None
        assert snap.bb_pct_b > 1.0

    def test_bb_pct_b_negative_when_price_below_lower(self, engine):
        """When the last close is below the lower band, pct_b should be < 0."""
        # Flat price of 100 → narrow BB bands
        n = 50
        closes = [100.0] * n
        df = make_ohlcv(closes)
        # Set last close well below 100 - any plausible lower band
        df.iloc[-1, df.columns.get_loc("close")] = 90.0
        df.iloc[-1, df.columns.get_loc("low")] = 89.5
        snap = engine.compute(df)
        assert snap.bb_pct_b is not None
        assert snap.bb_pct_b < 0.0


class TestMACDHistogram:
    def test_histogram_equals_line_minus_signal(self, engine, df_250):
        """macd_histogram ≈ macd_line - macd_signal."""
        snap = engine.compute(df_250)
        assert snap.macd_line is not None
        assert snap.macd_signal is not None
        assert snap.macd_histogram is not None
        expected = snap.macd_line - snap.macd_signal
        assert abs(snap.macd_histogram - expected) < 1e-6


class TestVWAP:
    def test_vwap_uses_df_column(self, engine):
        """When df has a 'vwap' column, snapshot.vwap should equal df['vwap'].iloc[-1]."""
        df = make_trending_up(250)
        # Manually add a specific vwap column
        df["vwap"] = df["close"] * 0.99   # deliberate offset to confirm we read the column
        snap = engine.compute(df)
        assert snap.vwap is not None
        assert abs(snap.vwap - float(df["vwap"].iloc[-1])) < 1e-9

    def test_vwap_is_none_when_column_missing(self, engine, df_250):
        """No vwap column → snapshot.vwap is None."""
        assert "vwap" not in df_250.columns
        snap = engine.compute(df_250)
        assert snap.vwap is None


class TestADXRange:
    def test_adx_between_0_and_100(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.adx is not None
        assert 0.0 <= snap.adx <= 100.0


class TestStochRSI:
    def test_stoch_rsi_range(self, engine, df_250):
        snap = engine.compute(df_250)
        assert snap.stoch_rsi_k is not None, "stoch_rsi_k should not be None with 250 bars"
        assert snap.stoch_rsi_d is not None, "stoch_rsi_d should not be None with 250 bars"
        assert 0.0 <= snap.stoch_rsi_k <= 100.0
        assert 0.0 <= snap.stoch_rsi_d <= 100.0


class TestNeverRaisesEdgeCases:
    def test_compute_with_none_input_never_raises(self):
        engine = IndicatorsEngine()
        result = engine.compute(None)  # must not raise
        assert result.ema_9 is None

    def test_compute_with_missing_required_columns_never_raises(self):
        engine = IndicatorsEngine()
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})  # missing high, low, volume
        result = engine.compute(df)  # must not raise
        assert result.ema_9 is None


# ── Contract: snapshot is frozen (immutable) ──────────────────────────────────

class TestImmutability:
    def test_snapshot_is_frozen(self, engine, df_250):
        snap = engine.compute(df_250)
        with pytest.raises((AttributeError, TypeError)):
            snap.rsi = 99.0  # type: ignore[misc]

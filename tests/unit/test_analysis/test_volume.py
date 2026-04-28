"""
Unit tests for VolumeEngine.

All DataFrames are built synthetically — no external data or fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from libs.analysis.volume.engine import VolumeContext, VolumeEngine
from libs.core.models.domain import SignalAction


# ── Helper ─────────────────────────────────────────────────────────────────────

def make_df(
    closes: list[float],
    volumes: list[float] | None = None,
    include_vwap: bool = False,
) -> pd.DataFrame:
    """
    Build a minimal enriched OHLCV DataFrame from close prices.

    Columns: open, high, low, close, volume, body_size, total_range, body_pct
    Optional: vwap (computed via cumulative VWAP formula)
    """
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    vols = np.array(volumes if volumes is not None else [1000.0] * n, dtype=float)
    opens = closes_arr * 0.999

    df = pd.DataFrame(
        {
            "open": opens,
            "high": closes_arr * 1.003,
            "low": closes_arr * 0.997,
            "close": closes_arr,
            "volume": vols,
        },
        index=pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC"),
    )

    df["body_size"] = (df["close"] - df["open"]).abs()
    df["total_range"] = df["high"] - df["low"]
    df["body_pct"] = df["body_size"] / df["total_range"].replace(0, np.nan)

    if include_vwap:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestRelativeVolume:
    """Test 1: High relative-volume spike is detected."""

    def test_high_rel_vol_spike(self) -> None:
        """Last bar volume spike should produce rel_vol > HIGH_REL_VOL threshold."""
        # 25 bars of normal volume (1 000), then a spike bar (3 000)
        base_vol = [1000.0] * 25
        spike_vol = base_vol + [3000.0]
        closes = [100.0] * 26

        df = make_df(closes, volumes=spike_vol)
        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df, SignalAction.BUY)

        assert ctx.relative_volume > VolumeEngine.HIGH_REL_VOL, (
            f"Expected rel_vol > {VolumeEngine.HIGH_REL_VOL}, got {ctx.relative_volume:.2f}"
        )


class TestDivergence:
    """Tests 2–3: Divergence detection."""

    def test_divergence_rising_close_falling_volume(self) -> None:
        """Rising close to a new high on falling volume should flag divergence."""
        # Prices rise to a new high on the last bar; volume drops
        closes = [100.0, 101.0, 102.0, 103.0, 104.0]
        # Volume is high on earlier bars, drops sharply on the last bar
        volumes = [1000.0, 1200.0, 1100.0, 900.0, 500.0]

        df = make_df(closes, volumes=volumes)
        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df, SignalAction.BUY)

        assert ctx.divergence == True, "Expected divergence=True on rising close with falling volume"

    def test_no_divergence_flat_bars(self) -> None:
        """Flat price with stable volume should not trigger divergence."""
        closes = [100.0] * 10
        volumes = [1000.0] * 10

        df = make_df(closes, volumes=volumes)
        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df, SignalAction.BUY)

        assert ctx.divergence == False, "Expected divergence=False on flat bars"


class TestEffortVsResult:
    """Test 4: High-effort-low-result detection."""

    def test_high_effort_low_result_detected(self) -> None:
        """
        High relative volume but tiny body (doji-like) should produce
        effort_vs_result == 'high_effort_low_result'.
        """
        # Build 25 bars of baseline volume so rolling avg is stable
        n_base = 25
        base_closes = [100.0] * n_base
        base_vols = [1000.0] * n_base
        df_base = make_df(base_closes, volumes=base_vols)

        # Last bar: volume spike but open ≈ close (doji body)
        # Manually craft the last row with a tiny body
        last_row = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.05],    # barely moved — tiny body
                "volume": [3000.0],   # 3× average → high effort
                "body_size": [0.05],
                "total_range": [2.0],
                "body_pct": [0.025],  # well below 0.3
            },
            index=pd.date_range("2024-01-02", periods=1, freq="1min", tz="UTC")
            + pd.Timedelta(minutes=n_base),
        )

        df = pd.concat([df_base, last_row])
        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df, SignalAction.BUY)

        assert ctx.effort_vs_result == "high_effort_low_result", (
            f"Expected 'high_effort_low_result', got '{ctx.effort_vs_result}'"
        )


class TestBreakoutConfirmed:
    """Test 5: Breakout confirmation."""

    def test_breakout_confirmed_high_vol_rising_trend_bullish_bar(self) -> None:
        """High vol + rising trend + bullish last bar should confirm breakout for BUY."""
        # Build increasing volumes (rising trend) culminating in a spike
        n = 25
        # Volume escalates: 800 → 1200, then spike at 2500
        vols = list(np.linspace(800, 1200, n - 1)) + [2500.0]
        # Rising closes
        closes = list(np.linspace(100, 105, n))

        df = make_df(closes, volumes=vols)

        # Make last bar explicitly bullish: close > open
        df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-1] * 0.998
        df.iloc[-1, df.columns.get_loc("body_size")] = (
            df["close"].iloc[-1] - df["open"].iloc[-1]
        )

        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df, SignalAction.BUY)

        assert ctx.breakout_confirmed is True, (
            f"Expected breakout_confirmed=True; rel_vol={ctx.relative_volume:.2f}, "
            f"trend={ctx.volume_trend}"
        )


class TestScoring:
    """Tests 6–7: Score computation."""

    def test_score_buy_with_breakout_above_threshold(self) -> None:
        """BUY signal with breakout_confirmed should yield score > 0.6."""
        n = 25
        vols = list(np.linspace(800, 1200, n - 1)) + [2500.0]
        closes = list(np.linspace(100, 105, n))

        df = make_df(closes, volumes=vols)
        df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-1] * 0.998

        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df, SignalAction.BUY)

        assert ctx.score > 0.6, f"Expected score > 0.6 with breakout; got {ctx.score:.3f}"

    def test_score_penalized_when_divergence_present(self) -> None:
        """Divergence should lower the score compared to a no-divergence baseline."""
        # Baseline: flat volume, flat price — no divergence
        closes_flat = [100.0] * 26
        vols_flat = [1000.0] * 26
        df_flat = make_df(closes_flat, volumes=vols_flat)

        engine = VolumeEngine(avg_period=20)
        ctx_flat = engine.analyze(df_flat, SignalAction.BUY)

        # Divergence case: new high close + sharply falling volume
        closes_div = [100.0, 101.0, 102.0, 103.0, 104.0]
        vols_div = [1000.0, 1200.0, 1100.0, 900.0, 500.0]
        df_div = make_df(closes_div, volumes=vols_div)
        ctx_div = engine.analyze(df_div, SignalAction.BUY)

        assert ctx_div.divergence == True, "Divergence case must have divergence=True"
        assert ctx_div.score < ctx_flat.score, (
            f"Divergence score ({ctx_div.score:.3f}) should be less than "
            f"baseline ({ctx_flat.score:.3f})"
        )


class TestEdgeCases:
    """Test 8: Empty DataFrame safety."""

    def test_empty_df_returns_default_context(self) -> None:
        """An empty DataFrame must not raise; result must have score == 0.5."""
        df = pd.DataFrame()
        engine = VolumeEngine(avg_period=20)

        ctx = engine.analyze(df, SignalAction.BUY)

        assert isinstance(ctx, VolumeContext)
        assert ctx.score == 0.5, f"Expected score=0.5 for empty df; got {ctx.score}"


class TestVwap:
    """Test 9: VWAP comparison."""

    def test_above_vwap_when_vwap_column_present(self) -> None:
        """above_vwap should reflect whether close > vwap in the last row."""
        # include_vwap computes cumulative VWAP; with rising closes the last
        # close will be above the cumulative VWAP.
        closes = list(np.linspace(100, 110, 25))
        df_above = make_df(closes, include_vwap=True)

        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df_above, SignalAction.BUY)

        # With rising prices, the last close is the highest so it will be
        # above the volume-weighted average of all bars.
        assert ctx.above_vwap is True, (
            f"Expected above_vwap=True for rising-price series; "
            f"close={df_above['close'].iloc[-1]:.2f}, "
            f"vwap={df_above['vwap'].iloc[-1]:.2f}"
        )

    def test_below_vwap_detected(self) -> None:
        """When close < vwap the flag should be False."""
        # Falling prices: vwap starts high, last close is the lowest bar
        closes = list(np.linspace(110, 100, 25))
        df_below = make_df(closes, include_vwap=True)

        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df_below, SignalAction.BUY)

        assert ctx.above_vwap is False, (
            f"Expected above_vwap=False for falling-price series; "
            f"close={df_below['close'].iloc[-1]:.2f}, "
            f"vwap={df_below['vwap'].iloc[-1]:.2f}"
        )


class TestVolumeTrend:
    """Test 10: Rising volume trend detection."""

    def test_rising_trend_on_increasing_volume_sequence(self) -> None:
        """Monotonically increasing volume should be classified as 'rising'."""
        closes = [100.0] * 10
        # Clearly increasing: each bar 10 % more than the previous
        volumes = [float(500 + i * 100) for i in range(10)]

        df = make_df(closes, volumes=volumes)
        engine = VolumeEngine(avg_period=20)
        ctx = engine.analyze(df, SignalAction.BUY)

        assert ctx.volume_trend == "rising", (
            f"Expected 'rising' trend; got '{ctx.volume_trend}'"
        )

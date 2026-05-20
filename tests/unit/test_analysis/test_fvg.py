"""
Unit tests for FVGDetector (Fair Value Gap detector).

Tests bullish/bearish FVG detection, filled-gap tracking, and graceful
handling of empty DataFrames.  All DataFrames are built synthetically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from libs.analysis.structure.fair_value_gap import FairValueGap, FVGDetector


# ── DataFrame helper ──────────────────────────────────────────────────────────

def _make_df(
    highs: list[float],
    lows: list[float],
    closes: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from highs and lows.

    closes defaults to midpoint of high/low per bar.
    open  = previous close (or midpoint for bar 0).
    volume defaults to 1000.0 per bar.
    """
    n = len(highs)
    assert len(lows) == n, "highs and lows must be the same length"

    highs_arr = np.array(highs, dtype=float)
    lows_arr = np.array(lows, dtype=float)

    if closes is None:
        closes_arr = (highs_arr + lows_arr) / 2.0
    else:
        assert len(closes) == n
        closes_arr = np.array(closes, dtype=float)

    opens_arr = np.empty(n, dtype=float)
    opens_arr[0] = closes_arr[0]
    opens_arr[1:] = closes_arr[:-1]

    if volumes is None:
        volumes_arr = np.full(n, 1_000.0)
    else:
        assert len(volumes) == n
        volumes_arr = np.array(volumes, dtype=float)

    return pd.DataFrame(
        {
            "open": opens_arr,
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
            "volume": volumes_arr,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min"),
    )


# ── Test 1: bullish FVG ───────────────────────────────────────────────────────

def test_bullish_fvg() -> None:
    """
    Three-candle pattern where candle[i-1].high < candle[i+1].low creates
    a bullish FVG (buying imbalance gap).

    Layout:
        bar 0 (prev): high=100, low=98   ← candle[i-1]
        bar 1 (mid):  high=103, low=99   ← driving impulse
        bar 2 (next): high=107, low=102  ← candle[i+1]

    Gap = (100, 102) — candle[i-1].high=100 < candle[i+1].low=102.
    """
    highs  = [100.0, 103.0, 107.0]
    lows   = [ 98.0,  99.0, 102.0]

    df = _make_df(highs, lows)
    detector = FVGDetector()
    fvgs = detector.detect(df)

    bullish = [f for f in fvgs if f.kind == "bullish_fvg"]
    assert len(bullish) >= 1, (
        f"Expected at least one bullish_fvg, got {fvgs}"
    )
    fvg = bullish[0]
    # Gap boundaries: lower=candle[i-1].high=100, upper=candle[i+1].low=102
    assert fvg.lower == pytest.approx(100.0, abs=1e-6)
    assert fvg.upper == pytest.approx(102.0, abs=1e-6)
    assert fvg.upper > fvg.lower
    assert fvg.size_pct > 0.0
    assert isinstance(fvg.kind, str)
    assert isinstance(fvg.filled, bool)


# ── Test 2: bearish FVG ───────────────────────────────────────────────────────

def test_bearish_fvg() -> None:
    """
    Three-candle pattern where candle[i-1].low > candle[i+1].high creates
    a bearish FVG (selling imbalance gap).

    Layout:
        bar 0 (prev): high=108, low=104  ← candle[i-1]
        bar 1 (mid):  high=105, low=100  ← driving impulse down
        bar 2 (next): high=102, low=96   ← candle[i+1]

    Gap = (102, 104) — candle[i-1].low=104 > candle[i+1].high=102.
    """
    highs  = [108.0, 105.0, 102.0]
    lows   = [104.0, 100.0,  96.0]

    df = _make_df(highs, lows)
    detector = FVGDetector()
    fvgs = detector.detect(df)

    bearish = [f for f in fvgs if f.kind == "bearish_fvg"]
    assert len(bearish) >= 1, (
        f"Expected at least one bearish_fvg, got {fvgs}"
    )
    fvg = bearish[0]
    # Gap boundaries: lower=candle[i+1].high=102, upper=candle[i-1].low=104
    assert fvg.lower == pytest.approx(102.0, abs=1e-6)
    assert fvg.upper == pytest.approx(104.0, abs=1e-6)
    assert fvg.upper > fvg.lower
    assert fvg.size_pct > 0.0


# ── Test 3: filled gap ────────────────────────────────────────────────────────

def test_filled_gap() -> None:
    """
    A bullish FVG that is subsequently revisited should be marked filled=True,
    while one that is not revisited should be marked filled=False.
    """
    # FVG at bars 0-2: gap = (100, 102)
    # bar 3: price drops back into gap → gap is filled
    highs_filled = [100.0, 103.0, 107.0,  103.0]
    lows_filled  = [ 98.0,  99.0, 102.0,   99.5]

    df_filled = _make_df(highs_filled, lows_filled)
    detector = FVGDetector()
    fvgs_filled = detector.detect(df_filled)

    bullish_filled = [f for f in fvgs_filled if f.kind == "bullish_fvg"]
    assert len(bullish_filled) >= 1, "Expected bullish_fvg in filled scenario"
    assert bullish_filled[0].filled is True, (
        f"Expected filled=True, got filled={bullish_filled[0].filled}"
    )

    # FVG at bars 0-2: gap = (100, 102)
    # bar 3: price continues up without revisiting gap → gap is NOT filled
    highs_unfilled = [100.0, 103.0, 107.0, 112.0]
    lows_unfilled  = [ 98.0,  99.0, 102.0, 108.0]

    df_unfilled = _make_df(highs_unfilled, lows_unfilled)
    fvgs_unfilled = detector.detect(df_unfilled)

    bullish_unfilled = [f for f in fvgs_unfilled if f.kind == "bullish_fvg"]
    assert len(bullish_unfilled) >= 1, "Expected bullish_fvg in unfilled scenario"
    assert bullish_unfilled[0].filled is False, (
        f"Expected filled=False, got filled={bullish_unfilled[0].filled}"
    )

    # Verify FairValueGap is immutable
    fvg = bullish_filled[0]
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        fvg.filled = False  # type: ignore[misc]


# ── Test 4: empty DataFrame ───────────────────────────────────────────────────

def test_empty_df() -> None:
    """
    Empty DataFrame must return an empty list without raising.
    """
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    detector = FVGDetector()
    result = detector.detect(df)

    assert result == [], f"Expected [], got {result}"

    # Also verify that a too-short DataFrame (< 3 bars) returns []
    df_short = _make_df([100.0, 101.0], [99.0, 100.0])
    result_short = detector.detect(df_short)
    assert result_short == [], f"Expected [] for 2-bar df, got {result_short}"

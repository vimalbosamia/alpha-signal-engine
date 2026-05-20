"""
Unit tests for OrderBlockDetector.

Tests bullish/bearish order block detection, field validation, and graceful
handling of empty DataFrames.  All DataFrames are built synthetically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataclasses import FrozenInstanceError

from libs.analysis.structure.order_blocks import OrderBlock, OrderBlockDetector


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


def _flat_bars(price: float, n: int) -> tuple[list[float], list[float], list[float]]:
    """Return n bars of flat price action (high+0.5, low-0.5, close=price)."""
    return (
        [price + 0.5] * n,
        [price - 0.5] * n,
        [price] * n,
    )


# ── Test 1: bullish order block ───────────────────────────────────────────────

def test_bullish_ob() -> None:
    """
    A bearish candle followed by a 3-bar bullish impulse > ATR×2 creates
    a bullish order block.  Zone upper = open, lower = low of that candle.
    """
    # 15 flat bars to build ATR baseline (~1.0 range each bar)
    flat_h, flat_l, flat_c = _flat_bars(100.0, 15)

    # Bearish OB candle: open=103, close=100 (bearish), low=99.5
    ob_bar_highs  = [103.0]
    ob_bar_lows   = [99.5]
    ob_bar_closes = [100.0]  # close < open → bearish
    ob_bar_opens  = [103.0]  # explicitly bearish

    # 3 bars of strong bullish impulse: high reaches ~110 (move > ATR×2 ≈ 2)
    impulse_highs  = [104.0, 107.0, 111.0]
    impulse_lows   = [101.0, 104.0, 108.0]
    impulse_closes = [103.5, 106.5, 110.5]

    highs  = flat_h  + ob_bar_highs  + impulse_highs
    lows   = flat_l  + ob_bar_lows   + impulse_lows
    closes = flat_c  + ob_bar_closes + impulse_closes

    # Patch opens so the OB candle is clearly bearish
    df = _make_df(highs, lows, closes)
    # Manually override opens for the OB candle (bar index 15)
    opens_arr = df["open"].to_numpy(dtype=float)
    opens_arr[15] = 103.0
    df = df.assign(open=opens_arr)

    detector = OrderBlockDetector(atr_factor=2.0, impulse_bars=3)
    obs = detector.detect(df)

    bullish = [ob for ob in obs if ob.kind == "bullish_ob"]
    assert len(bullish) >= 1, (
        f"Expected at least one bullish_ob, got {obs}"
    )
    ob = bullish[0]
    assert ob.upper > ob.lower, "OB upper must be above lower"
    assert ob.strength > 0.0
    assert isinstance(ob.explanation, str) and len(ob.explanation) > 0


# ── Test 2: bearish order block ───────────────────────────────────────────────

def test_bearish_ob() -> None:
    """
    A bullish candle followed by a 3-bar bearish impulse > ATR×2 creates
    a bearish order block.  Zone upper = high, lower = close of that candle.
    """
    # 15 flat bars to build ATR baseline (~1.0 range each bar)
    flat_h, flat_l, flat_c = _flat_bars(100.0, 15)

    # Bullish OB candle: open=97, close=100 (bullish), high=100.5
    ob_bar_highs  = [100.5]
    ob_bar_lows   = [97.0]
    ob_bar_closes = [100.0]  # close > open → bullish

    # 3 bars of strong bearish impulse: low drops to ~88 (move > ATR×2 ≈ 2)
    impulse_highs  = [99.0, 96.0, 92.0]
    impulse_lows   = [96.0, 93.0, 88.5]
    impulse_closes = [96.5, 93.5, 89.0]

    highs  = flat_h  + ob_bar_highs  + impulse_highs
    lows   = flat_l  + ob_bar_lows   + impulse_lows
    closes = flat_c  + ob_bar_closes + impulse_closes

    df = _make_df(highs, lows, closes)
    # Override opens for OB candle to be clearly bullish (open < close)
    opens_arr = df["open"].to_numpy(dtype=float)
    opens_arr[15] = 97.0
    df = df.assign(open=opens_arr)

    detector = OrderBlockDetector(atr_factor=2.0, impulse_bars=3)
    obs = detector.detect(df)

    bearish = [ob for ob in obs if ob.kind == "bearish_ob"]
    assert len(bearish) >= 1, (
        f"Expected at least one bearish_ob, got {obs}"
    )
    ob = bearish[0]
    assert ob.upper > ob.lower, "OB upper must be above lower"
    assert ob.strength > 0.0
    assert isinstance(ob.explanation, str) and len(ob.explanation) > 0


# ── Test 3: order block field validation ──────────────────────────────────────

def test_ob_fields() -> None:
    """
    All fields of detected OrderBlock must have the correct types and ranges.
    OrderBlock must be immutable (frozen dataclass).
    """
    flat_h, flat_l, flat_c = _flat_bars(100.0, 15)

    # One bullish OB scenario (same as test_bullish_ob)
    highs  = flat_h  + [103.0, 104.0, 107.0, 111.0]
    lows   = flat_l  + [99.5,  101.0, 104.0, 108.0]
    closes = flat_c  + [100.0, 103.5, 106.5, 110.5]

    df = _make_df(highs, lows, closes)
    opens_arr = df["open"].to_numpy(dtype=float)
    opens_arr[15] = 103.0
    df = df.assign(open=opens_arr)

    detector = OrderBlockDetector(atr_factor=2.0, impulse_bars=3)
    obs = detector.detect(df)

    assert isinstance(obs, list), "detect() must return a list"

    for ob in obs:
        assert isinstance(ob, OrderBlock)
        assert ob.kind in ("bullish_ob", "bearish_ob"), f"Unexpected kind: {ob.kind!r}"
        assert isinstance(ob.upper, float)
        assert isinstance(ob.lower, float)
        assert ob.upper >= ob.lower, "upper must be >= lower"
        assert isinstance(ob.strength, float)
        assert 0.0 <= ob.strength <= 1.0, f"strength out of range: {ob.strength}"
        assert isinstance(ob.index, int)
        assert isinstance(ob.explanation, str) and len(ob.explanation) > 0

    # Verify immutability
    if obs:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            obs[0].kind = "bullish_ob"  # type: ignore[misc]


# ── Test 4: empty DataFrame ───────────────────────────────────────────────────

def test_empty_df() -> None:
    """
    Empty DataFrame must return an empty list without raising.
    """
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    detector = OrderBlockDetector()
    result = detector.detect(df)

    assert result == [], f"Expected [], got {result}"

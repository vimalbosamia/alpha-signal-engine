"""
Unit tests for SwingPointDetector.

All DataFrames are built synthetically — no external data or fixtures.
Tests follow RED → GREEN → REFACTOR TDD cycle.
"""
from __future__ import annotations

import pandas as pd
import pytest

from libs.analysis.structure.swing_points import SwingPoint, SwingPointDetector, classify_trend


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from explicit high/low series.

    open  = (high + low) / 2
    close = (high + low) / 2
    volume = 1000
    """
    assert len(highs) == len(lows), "highs and lows must be same length"
    n = len(highs)
    mid = [(h + lo) / 2.0 for h, lo in zip(highs, lows)]
    return pd.DataFrame(
        {
            "open": mid,
            "high": highs,
            "low": lows,
            "close": mid,
            "volume": [1000] * n,
        },
        index=pd.RangeIndex(n),
    )


# ── Test 1: detect swing high ─────────────────────────────────────────────────

def test_detects_swing_high() -> None:
    """
    A single peak at index 2 (lookback=2) should be detected as a swing high
    with price 15.
    """
    highs = [10.0, 11.0, 15.0, 12.0, 10.0]
    lows  = [8.0,  9.0,  13.0, 9.0,  8.0]
    df = _make_df(highs, lows)

    detector = SwingPointDetector(lookback=2)
    points = detector.detect(df)

    swing_highs = [p for p in points if p.kind == "high"]
    assert len(swing_highs) >= 1, "Expected at least one swing high"
    prices = {p.price for p in swing_highs}
    assert 15.0 in prices, f"Expected swing high at price 15.0, got {prices}"


# ── Test 2: detect swing low ──────────────────────────────────────────────────

def test_detects_swing_low() -> None:
    """
    A single trough at index 2 should be detected as a swing low with price 5.
    """
    highs = [15.0, 12.0, 8.0,  12.0, 15.0]
    lows  = [13.0, 10.0, 5.0,  10.0, 13.0]
    df = _make_df(highs, lows)

    detector = SwingPointDetector(lookback=2)
    points = detector.detect(df)

    swing_lows = [p for p in points if p.kind == "low"]
    assert len(swing_lows) >= 1, "Expected at least one swing low"
    prices = {p.price for p in swing_lows}
    assert 5.0 in prices, f"Expected swing low at price 5.0, got {prices}"


# ── Test 3: higher high classification ───────────────────────────────────────

def test_higher_high_classification() -> None:
    """
    Two swing highs where the second is higher than the first → second is "HH".

    Data pattern (lookback=1):
      highs: flat, peak1=15, flat, flat, peak2=18, flat
    """
    highs = [10.0, 15.0, 10.0, 10.0, 18.0, 10.0]
    lows  = [8.0,  12.0,  8.0,  8.0, 15.0,  8.0]
    df = _make_df(highs, lows)

    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)

    swing_highs = [p for p in points if p.kind == "high"]
    assert len(swing_highs) >= 2, (
        f"Expected at least 2 swing highs, got {len(swing_highs)}: {swing_highs}"
    )
    # Sort by index and check second classification
    swing_highs_sorted = sorted(swing_highs, key=lambda p: p.index)
    second_high = swing_highs_sorted[1]
    assert second_high.classification == "HH", (
        f"Expected 'HH' for second swing high (price={second_high.price}), "
        f"got '{second_high.classification}'"
    )


# ── Test 4: lower low classification ─────────────────────────────────────────

def test_lower_low_classification() -> None:
    """
    Two swing lows where the second is lower than the first → second is "LL".

    Data pattern (lookback=1):
      lows: flat, trough1=8, flat, flat, trough2=5, flat
    """
    highs = [15.0, 12.0, 15.0, 15.0, 10.0, 15.0]
    lows  = [12.0,  8.0, 12.0, 12.0,  5.0, 12.0]
    df = _make_df(highs, lows)

    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)

    swing_lows = [p for p in points if p.kind == "low"]
    assert len(swing_lows) >= 2, (
        f"Expected at least 2 swing lows, got {len(swing_lows)}: {swing_lows}"
    )
    swing_lows_sorted = sorted(swing_lows, key=lambda p: p.index)
    second_low = swing_lows_sorted[1]
    assert second_low.classification == "LL", (
        f"Expected 'LL' for second swing low (price={second_low.price}), "
        f"got '{second_low.classification}'"
    )


# ── Test 5: uptrend pattern ───────────────────────────────────────────────────

def test_uptrend_pattern() -> None:
    """
    A sequence of HH+HL swing points should yield classify_trend "uptrend" or "neutral".

    Build a rising staircase: each peak higher than previous, each trough higher
    than previous trough.
    """
    highs = [
        8.0, 12.0, 8.0,    # swing high at idx 1, price 12
        9.0, 15.0, 9.0,    # swing high at idx 4, price 15  → HH
        10.0, 18.0, 10.0,  # swing high at idx 7, price 18  → HH
        11.0, 21.0, 11.0,  # swing high at idx 10, price 21 → HH
    ]
    lows = [
        6.0, 9.0,  6.0,    # swing low at idx 0/2
        7.0, 11.0, 7.0,    # swing low at idx 3/5, price 7  → HL
        8.0, 14.0, 8.0,    # swing low at idx 6/8, price 8  → HL
        9.0, 18.0, 9.0,    # swing low at idx 9/11, price 9 → HL
    ]
    df = _make_df(highs, lows)

    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)

    result = classify_trend(points)
    assert result in ("uptrend", "neutral"), (
        f"Expected 'uptrend' or 'neutral', got '{result}'. Points: {points}"
    )


# ── Test 6: downtrend pattern ─────────────────────────────────────────────────

def test_downtrend_pattern() -> None:
    """
    A sequence of LH+LL swing points should yield classify_trend "downtrend" or "neutral".

    Build a falling staircase: each peak lower than previous, each trough lower
    than previous trough.
    """
    highs = [
        20.0, 25.0, 20.0,   # swing high at idx 1, price 25
        18.0, 22.0, 18.0,   # swing high at idx 4, price 22 → LH
        16.0, 19.0, 16.0,   # swing high at idx 7, price 19 → LH
        14.0, 16.0, 14.0,   # swing high at idx 10, price 16 → LH
    ]
    lows = [
        17.0, 22.0, 17.0,   # swing low at idx 0/2, price 17
        15.0, 19.0, 15.0,   # swing low at idx 3/5, price 15 → LL
        13.0, 16.0, 13.0,   # swing low at idx 6/8, price 13 → LL
        11.0, 13.0, 11.0,   # swing low at idx 9/11, price 11 → LL
    ]
    df = _make_df(highs, lows)

    detector = SwingPointDetector(lookback=1)
    points = detector.detect(df)

    result = classify_trend(points)
    assert result in ("downtrend", "neutral"), (
        f"Expected 'downtrend' or 'neutral', got '{result}'. Points: {points}"
    )


# ── Test 7: empty DataFrame ───────────────────────────────────────────────────

def test_empty_df() -> None:
    """Empty DataFrame must return an empty list without raising."""
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    detector = SwingPointDetector(lookback=2)
    points = detector.detect(df)

    assert points == [], f"Expected [], got {points}"


# ── Test 8: lookback parameter ────────────────────────────────────────────────

def test_lookback_parameter() -> None:
    """
    lookback=1 should detect more peaks than lookback=2 on the same data,
    because lookback=2 requires a wider window around each candidate bar.
    """
    # Build data with several local peaks and troughs
    highs = [10.0, 14.0, 11.0, 16.0, 12.0, 18.0, 13.0, 15.0, 11.0]
    lows  = [8.0,  11.0, 9.0,  13.0, 9.0,  15.0, 10.0, 12.0, 9.0]
    df = _make_df(highs, lows)

    points_lb1 = SwingPointDetector(lookback=1).detect(df)
    points_lb2 = SwingPointDetector(lookback=2).detect(df)

    # lookback=1 sees more (or equal) swing points than lookback=2
    assert len(points_lb1) >= len(points_lb2), (
        f"lookback=1 should detect >= points vs lookback=2. "
        f"Got lb1={len(points_lb1)}, lb2={len(points_lb2)}"
    )
    # Additionally, lookback=2 must detect fewer OR equal peaks (stricter)
    highs_lb1 = len([p for p in points_lb1 if p.kind == "high"])
    highs_lb2 = len([p for p in points_lb2 if p.kind == "high"])
    assert highs_lb1 >= highs_lb2, (
        f"lookback=1 should detect >= swing highs vs lookback=2. "
        f"Got lb1={highs_lb1}, lb2={highs_lb2}"
    )

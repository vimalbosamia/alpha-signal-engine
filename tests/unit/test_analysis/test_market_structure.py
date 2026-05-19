"""
Unit tests for MarketStructureAnalyzer.

Tests BOS (Break of Structure) and CHoCH (Change of Character) detection.
All DataFrames are built synthetically — no external data or fixtures.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Minimal stub for SwingPoint / SwingPointDetector / classify_trend
# Used as a fallback if swing_points.py doesn't exist yet.
# ---------------------------------------------------------------------------
try:
    from libs.analysis.structure.swing_points import (
        SwingPoint,
        SwingPointDetector,
        classify_trend,
    )
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)  # type: ignore[no-redef]
    class SwingPoint:
        index: int
        price: float
        kind: str          # "high" or "low"
        classification: str  # "HH", "HL", "LH", "LL", or "pivot_high"/"pivot_low"

    class SwingPointDetector:  # type: ignore[no-redef]
        def __init__(self, lookback: int = 2) -> None:
            self.lookback = lookback

        def detect(self, df: pd.DataFrame) -> list[SwingPoint]:
            if df.empty or len(df) < 3:
                return []
            points: list[SwingPoint] = []
            highs = df["high"].to_numpy()
            lows = df["low"].to_numpy()
            n = len(highs)
            lb = self.lookback

            raw_highs: list[tuple[int, float]] = []
            raw_lows: list[tuple[int, float]] = []
            for i in range(lb, n - lb):
                window_h = highs[i - lb: i + lb + 1]
                window_l = lows[i - lb: i + lb + 1]
                if highs[i] == window_h.max():
                    raw_highs.append((i, float(highs[i])))
                if lows[i] == window_l.min():
                    raw_lows.append((i, float(lows[i])))

            for k, (idx, price) in enumerate(raw_highs):
                if k == 0:
                    label = "pivot_high"
                else:
                    label = "HH" if price > raw_highs[k - 1][1] else "LH"
                points.append(SwingPoint(index=idx, price=price, kind="high", classification=label))

            for k, (idx, price) in enumerate(raw_lows):
                if k == 0:
                    label = "pivot_low"
                else:
                    label = "HL" if price > raw_lows[k - 1][1] else "LL"
                points.append(SwingPoint(index=idx, price=price, kind="low", classification=label))

            return sorted(points, key=lambda p: p.index)

    def classify_trend(points: list[SwingPoint]) -> str:  # type: ignore[no-redef]
        highs = [p for p in points if p.kind == "high" and p.classification in ("HH", "LH")]
        lows = [p for p in points if p.kind == "low" and p.classification in ("HL", "LL")]
        if len(highs) < 2 or len(lows) < 2:
            return "neutral"
        last_h = highs[-1].classification
        prev_h = highs[-2].classification
        last_l = lows[-1].classification
        prev_l = lows[-2].classification
        if last_h == "HH" and prev_h == "HH" and last_l == "HL" and prev_l == "HL":
            return "bullish"
        if last_h == "LH" and prev_h == "LH" and last_l == "LL" and prev_l == "LL":
            return "bearish"
        return "neutral"


from libs.analysis.structure.market_structure import (
    MarketStructureAnalyzer,
    StructureAnalysis,
    StructureEvent,
)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_df(
    highs: list[float],
    lows: list[float],
    closes: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from highs and lows.

    closes defaults to midpoint of high/low per bar.
    open  = previous close (or midpoint for bar 0).
    volume = 1000.0 per bar.
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

    opens = np.empty(n, dtype=float)
    opens[0] = closes_arr[0]
    opens[1:] = closes_arr[:-1]

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs_arr,
            "low": lows_arr,
            "close": closes_arr,
            "volume": np.full(n, 1_000.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min"),
    )


# ── Test 1: BOS bullish ────────────────────────────────────────────────────────

def test_bos_bullish() -> None:
    """
    Price breaks above a previous swing high → a bullish BOS event is detected.

    We construct a clear scenario:
      - Several bars establishing a swing high near 110
      - Then a close that punches above that level
    """
    # Build a clear sequence:
    # bars 0–9: price oscillates around 100, building a swing high at ~110
    # bars 10+: price surges to 115 (breaks the 110 swing high)
    highs =  [100, 102, 108, 110, 107, 104, 102, 105, 108, 110,  115, 116]
    lows =   [ 98,  99, 100, 102, 100,  99,  98,  99, 100, 101,  110, 111]
    closes = [ 99, 101, 104, 106, 103, 101,  99, 103, 105, 105,  114, 115]

    df = _make_df(highs, lows, closes)
    analyzer = MarketStructureAnalyzer(lookback=2)
    result = analyzer.analyze(df)

    bos_bullish = [e for e in result.events if e.kind == "BOS" and e.direction == "bullish"]
    assert len(bos_bullish) >= 1, (
        f"Expected at least one bullish BOS event, got events={result.events}"
    )
    event = bos_bullish[-1]
    assert event.broken_level > 0
    assert event.price > event.broken_level
    assert isinstance(event.explanation, str) and len(event.explanation) > 0


# ── Test 2: BOS bearish ────────────────────────────────────────────────────────

def test_bos_bearish() -> None:
    """
    Price breaks below a previous swing low → a bearish BOS event is detected.

    We construct a sequence where a swing low is established near 90,
    then a close punches below it.
    """
    highs =  [104, 102, 100,  98,  96,  97,  98,  97,  96,  95,   90,  89]
    lows =   [100,  98,  95,  93,  91,  92,  93,  92,  91,  90,   85,  84]
    closes = [102, 100,  97,  95,  93,  94,  95,  94,  93,  92,   86,  85]

    df = _make_df(highs, lows, closes)
    analyzer = MarketStructureAnalyzer(lookback=2)
    result = analyzer.analyze(df)

    bos_bearish = [e for e in result.events if e.kind == "BOS" and e.direction == "bearish"]
    assert len(bos_bearish) >= 1, (
        f"Expected at least one bearish BOS event, got events={result.events}"
    )
    event = bos_bearish[-1]
    assert event.broken_level > 0
    assert event.price < event.broken_level
    assert isinstance(event.explanation, str) and len(event.explanation) > 0


# ── Test 3: CHoCH bullish ─────────────────────────────────────────────────────

def test_choch_bullish() -> None:
    """
    Downtrend (LH+LL pattern) followed by a bullish BOS above a swing high
    → that BOS should be classified as CHoCH (Change of Character).
    """
    # First, build a clear downtrend: LH + LL pattern
    # Swing highs: 110, 108, 106  (Lower Highs)
    # Swing lows:  100,  97,  94  (Lower Lows)
    # Then a bullish surge to 112 that breaks above the last swing high (106)

    n = 30
    highs:  list[float] = []
    lows:   list[float] = []
    closes: list[float] = []

    # Phase 1: downtrend (bars 0–23)
    swing_highs = [110.0, 108.0, 106.0]
    swing_lows  = [100.0,  97.0,  94.0]

    price = 105.0
    for i, (sh, sl) in enumerate(zip(swing_highs, swing_lows)):
        # 4 bars going up to swing high
        for j in range(4):
            frac = (j + 1) / 4.0
            h = price + (sh - price) * frac
            l = h - 2.0
            c = (h + l) / 2.0
            highs.append(round(h, 2))
            lows.append(round(l, 2))
            closes.append(round(c, 2))
        # 4 bars going down to swing low
        for j in range(4):
            frac = (j + 1) / 4.0
            h = sh - (sh - sl) * frac * 0.5
            l = h - 2.0
            c = (h + l) / 2.0
            highs.append(round(h, 2))
            lows.append(round(l, 2))
            closes.append(round(c, 2))
        price = sl

    # Phase 2: bullish surge breaking above last swing high (106) → CHoCH
    for j in range(6):
        h = 94.0 + (j + 1) * 3.5
        l = h - 1.5
        c = h - 0.5
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        closes.append(round(c, 2))

    df = _make_df(highs, lows, closes)
    analyzer = MarketStructureAnalyzer(lookback=2)
    result = analyzer.analyze(df)

    choch_events = [e for e in result.events if e.kind == "CHoCH"]
    assert len(choch_events) >= 1, (
        f"Expected at least one CHoCH event after downtrend+breakout.\n"
        f"Events: {result.events}\n"
        f"trend_bias: {result.trend_bias}"
    )
    bullish_choch = [e for e in choch_events if e.direction == "bullish"]
    assert len(bullish_choch) >= 1, (
        f"Expected bullish CHoCH, got: {choch_events}"
    )


# ── Test 4: result fields ─────────────────────────────────────────────────────

def test_structure_result_fields() -> None:
    """
    StructureAnalysis must expose all documented fields with correct types.
    """
    highs =  [100, 102, 105, 104, 103, 106, 108]
    lows =   [ 97,  99, 101, 100,  99, 102, 104]

    df = _make_df(highs, lows)
    analyzer = MarketStructureAnalyzer(lookback=2)
    result = analyzer.analyze(df)

    assert isinstance(result, StructureAnalysis)
    assert result.trend_bias in ("bullish", "bearish", "neutral")
    assert isinstance(result.swing_points, list)
    assert isinstance(result.events, list)
    assert isinstance(result.strength, float)
    assert 0.0 <= result.strength <= 1.0
    assert isinstance(result.explanation, str)

    # last_swing_high and last_swing_low can be float or None
    assert result.last_swing_high is None or isinstance(result.last_swing_high, float)
    assert result.last_swing_low is None or isinstance(result.last_swing_low, float)

    # All events are StructureEvent instances
    for event in result.events:
        assert isinstance(event, StructureEvent)
        assert event.kind in ("BOS", "CHoCH")
        assert event.direction in ("bullish", "bearish")
        assert isinstance(event.index, int)
        assert isinstance(event.price, float)
        assert isinstance(event.broken_level, float)
        assert isinstance(event.explanation, str)

    # StructureAnalysis is frozen (immutable)
    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        result.trend_bias = "neutral"  # type: ignore[misc]

    # StructureEvent is frozen (immutable)
    if result.events:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            result.events[0].kind = "BOS"  # type: ignore[misc]


# ── Test 5: empty DataFrame ───────────────────────────────────────────────────

def test_empty_df() -> None:
    """
    Empty DataFrame → neutral trend_bias, empty events list, no exception raised.
    """
    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    analyzer = MarketStructureAnalyzer(lookback=2)
    result = analyzer.analyze(df)

    assert result.trend_bias == "neutral"
    assert result.events == []
    assert result.swing_points == []
    assert result.strength == 0.0
    assert result.last_swing_high is None
    assert result.last_swing_low is None

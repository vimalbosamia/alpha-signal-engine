"""
Unit tests for MarketStructureEngine.

All DataFrames are built synthetically — no external data or fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from libs.analysis.structure.engine import MarketStructure, MarketStructureEngine, StructurePoint
from libs.core.models.domain import TrendDirection


# ── Helper ────────────────────────────────────────────────────────────────────

def make_df(closes: list[float]) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from a list of close prices.

    high  = close * 1.002
    low   = close * 0.998
    open  = close * 0.999
    volume = 1000.0 per bar
    """
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    df = pd.DataFrame(
        {
            "open": closes_arr * 0.999,
            "high": closes_arr * 1.002,
            "low": closes_arr * 0.998,
            "close": closes_arr,
            "volume": [1000.0] * n,
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min"),
    )
    return df


# ── Uptrend detection ─────────────────────────────────────────────────────────

def test_uptrend_detection() -> None:
    """
    Clear HH/HL pattern over 60 bars should be classified as UPTREND.

    Pattern: price rises with each cycle, and each trough is higher than
    the previous trough.
    """
    # Build 60 bars: slow staircase up with oscillations
    # cycle: go up 5 bars, dip 3 bars, each cycle starts higher
    closes: list[float] = []
    base = 100.0
    for cycle in range(8):
        peak = base + (cycle + 1) * 4.0
        trough = base + cycle * 4.0 + 1.0   # trough is higher than previous cycle base
        # 5 bars rising to peak
        closes.extend(np.linspace(base + cycle * 4.0, peak, 5).tolist())
        # 3 bars falling to trough
        closes.extend(np.linspace(peak, trough, 3).tolist())
        base = trough

    # Trim / pad to exactly 60 bars
    closes = closes[:60] if len(closes) >= 60 else closes + [closes[-1]] * (60 - len(closes))

    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=3, lookback=60)
    result = engine.analyze(df)

    assert result.trend == TrendDirection.UPTREND, (
        f"Expected UPTREND, got {result.trend}. "
        f"Points: {[(p.point_type, round(p.price, 2)) for p in result.points]}"
    )
    assert result.trend_strength > 0.0
    assert result.swing_high is not None
    assert result.swing_low is not None


# ── Downtrend detection ───────────────────────────────────────────────────────

def test_downtrend_detection() -> None:
    """
    Clear LH/LL pattern over 60 bars should be classified as DOWNTREND.
    """
    closes: list[float] = []
    base = 200.0
    for cycle in range(8):
        trough = base - (cycle + 1) * 4.0
        peak = base - cycle * 4.0 - 1.0   # peak is lower than previous cycle base
        # 5 bars falling to trough
        closes.extend(np.linspace(base - cycle * 4.0, trough, 5).tolist())
        # 3 bars bouncing to lower peak
        closes.extend(np.linspace(trough, peak, 3).tolist())
        base = peak

    closes = closes[:60] if len(closes) >= 60 else closes + [closes[-1]] * (60 - len(closes))

    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=3, lookback=60)
    result = engine.analyze(df)

    assert result.trend == TrendDirection.DOWNTREND, (
        f"Expected DOWNTREND, got {result.trend}. "
        f"Points: {[(p.point_type, round(p.price, 2)) for p in result.points]}"
    )
    assert result.trend_strength > 0.0
    assert result.swing_high is not None
    assert result.swing_low is not None


# ── Ranging detection ─────────────────────────────────────────────────────────

def test_ranging_detection() -> None:
    """
    Build a series where pivot highs alternate HH / LH and pivot lows
    alternate HL / LL.  The last two highs are HH + LH (mixed) and the last
    two lows are LL + HL (mixed), so neither UPTREND nor DOWNTREND conditions
    are fully satisfied and the result must be RANGING or UNKNOWN.

    We use sharp V/inverted-V shapes (1 bar peak / 1 bar trough) so that
    pivot_strength=1 picks exactly one pivot per swing, avoiding duplicate
    labelling caused by flat tops in linspace segments.

    Swing highs (raw close values):  102, 105, 103, 107, 104
    Swing lows  (raw close values):  98, 96, 99, 95, 97

    High labels (relative to previous high):
      105 > 102 → HH
      103 < 105 → LH
      107 > 103 → HH
      104 < 107 → LH   ← last two highs: HH, LH  (mixed)

    Low labels  (relative to previous low):
       96 < 98  → LL
       99 > 96  → HL
       95 < 99  → LL
       97 > 95  → HL   ← last two lows: LL, HL   (mixed)

    Mixed → RANGING (last two highs disagree with last two lows).
    """
    # Each swing: 4 bars rising to peak, 1 bar at peak, 4 bars falling to trough, 1 bar at trough.
    swing_peaks   = [102.0, 105.0, 103.0, 107.0, 104.0]
    swing_troughs = [98.0,   96.0,  99.0,  95.0,  97.0]

    closes: list[float] = [100.0]  # starting bar

    for k in range(len(swing_peaks)):
        prev = closes[-1]
        peak = swing_peaks[k]
        trough = swing_troughs[k]

        # 3 rising bars + 1 peak bar
        closes.extend(np.linspace(prev, peak, 3).tolist())
        closes.append(peak)  # isolated peak
        # 3 falling bars + 1 trough bar
        closes.extend(np.linspace(peak, trough, 3).tolist())
        closes.append(trough)  # isolated trough

    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=1, lookback=len(closes))
    result = engine.analyze(df)

    assert result.trend in (TrendDirection.RANGING, TrendDirection.UNKNOWN), (
        f"Expected RANGING or UNKNOWN, got {result.trend}. "
        f"Points: {[(p.point_type, round(p.price, 2)) for p in result.points]}"
    )


# ── Break of Structure — uptrend ──────────────────────────────────────────────

def test_bos_in_uptrend() -> None:
    """
    After establishing an uptrend, a final close well below the swing low
    should trigger break_of_structure=True.
    """
    # Build an uptrend for the first 50 bars
    closes: list[float] = []
    base = 100.0
    for cycle in range(7):
        peak = base + (cycle + 1) * 4.0
        trough = base + cycle * 4.0 + 1.0
        closes.extend(np.linspace(base + cycle * 4.0, peak, 5).tolist())
        closes.extend(np.linspace(peak, trough, 3).tolist())
        base = trough

    closes = closes[:50]

    # The swing_low will be somewhere around 100–105 range.
    # Add a sharp drop well below 100 to trigger BoS.
    closes.extend([90.0, 88.0, 85.0, 83.0, 80.0, 78.0, 75.0, 73.0, 70.0, 68.0])

    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=3, lookback=len(closes))
    result = engine.analyze(df)

    # The engine should either still classify uptrend (BoS against it) or ranging.
    # Key assertion: if uptrend was established, BoS must be True.
    if result.trend == TrendDirection.UPTREND:
        assert result.break_of_structure is True, (
            f"Expected BoS=True in uptrend after sharp drop. "
            f"swing_low={result.swing_low}, last_close={closes[-1]}"
        )
    else:
        # If the drop shifted trend classification itself, BoS may be False.
        # Just verify no exception was raised.
        assert isinstance(result.break_of_structure, bool)


def test_no_bos_in_clean_uptrend() -> None:
    """
    A clean uptrend with no reversal should have break_of_structure=False.
    """
    closes = list(np.linspace(100.0, 160.0, 60))
    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=3, lookback=60)
    result = engine.analyze(df)

    # Either uptrend or unknown (a perfect linear series has no pivots)
    assert result.break_of_structure is False


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_dataframe_returns_unknown() -> None:
    """Empty DataFrame must return UNKNOWN without raising."""
    df = pd.DataFrame()
    engine = MarketStructureEngine()
    result = engine.analyze(df)

    assert result.trend == TrendDirection.UNKNOWN
    assert result.points == []
    assert result.break_of_structure is False
    assert result.swing_high is None
    assert result.swing_low is None


def test_too_few_bars_returns_unknown() -> None:
    """
    Fewer than 2 * pivot_strength + 1 bars should return UNKNOWN without raising.
    """
    pivot_strength = 5
    # Need at least 11 bars; supply only 8
    closes = [100.0 + i for i in range(8)]
    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=pivot_strength)
    result = engine.analyze(df)

    assert result.trend == TrendDirection.UNKNOWN
    assert isinstance(result.notes, str) and len(result.notes) > 0


def test_exactly_minimum_bars_does_not_raise() -> None:
    """
    Exactly 2 * pivot_strength + 1 bars should not raise (may return UNKNOWN).
    """
    pivot_strength = 5
    min_bars = 2 * pivot_strength + 1  # = 11
    closes = [100.0 + i * 0.1 for i in range(min_bars)]
    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=pivot_strength)
    result = engine.analyze(df)  # must not raise

    assert isinstance(result, MarketStructure)
    assert result.trend in list(TrendDirection)


# ── Structure output integrity ────────────────────────────────────────────────

def test_structure_points_are_immutable() -> None:
    """StructurePoint objects must be frozen (immutable)."""
    sp = StructurePoint(idx=0, price=100.0, point_type="HH")
    with pytest.raises(Exception):
        sp.price = 200.0  # type: ignore[misc]


def test_market_structure_is_immutable() -> None:
    """MarketStructure must be frozen."""
    ms = MarketStructure(
        trend=TrendDirection.RANGING,
        points=[],
        trend_strength=0.0,
        break_of_structure=False,
        swing_high=None,
        swing_low=None,
    )
    with pytest.raises(Exception):
        ms.trend = TrendDirection.UPTREND  # type: ignore[misc]


def test_trend_strength_bounds() -> None:
    """trend_strength must always be in [0.0, 1.0]."""
    closes: list[float] = []
    base = 100.0
    for cycle in range(8):
        peak = base + (cycle + 1) * 4.0
        trough = base + cycle * 4.0 + 1.0
        closes.extend(np.linspace(base + cycle * 4.0, peak, 5).tolist())
        closes.extend(np.linspace(peak, trough, 3).tolist())
        base = trough

    closes = closes[:60]
    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=3, lookback=60)
    result = engine.analyze(df)

    assert 0.0 <= result.trend_strength <= 1.0


def test_swing_high_geq_swing_low() -> None:
    """When both swing_high and swing_low are set, high >= low must hold."""
    closes: list[float] = []
    base = 100.0
    for cycle in range(8):
        peak = base + (cycle + 1) * 4.0
        trough = base + cycle * 4.0 + 1.0
        closes.extend(np.linspace(base + cycle * 4.0, peak, 5).tolist())
        closes.extend(np.linspace(peak, trough, 3).tolist())
        base = trough

    closes = closes[:60]
    df = make_df(closes)
    engine = MarketStructureEngine(pivot_strength=3, lookback=60)
    result = engine.analyze(df)

    if result.swing_high is not None and result.swing_low is not None:
        assert result.swing_high >= result.swing_low

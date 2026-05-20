"""
FVGDetector — identifies Fair Value Gaps (imbalances) from OHLCV data.

Fair Value Gaps are 3-candle patterns where the market moves so fast that
price leaves an unfilled gap.  Institutions often seek to rebalance these gaps.

Bullish FVG: candle[i-1].high < candle[i+1].low
             → gap from candle[i-1].high to candle[i+1].low (buying imbalance)

Bearish FVG: candle[i-1].low > candle[i+1].high
             → gap from candle[i+1].high to candle[i-1].low (selling imbalance)

A gap is "filled" when subsequent price action returns to trade within the gap.

Design rules:
  - Never raises on empty or short DataFrames — returns [] safely.
  - All output objects are frozen dataclasses (immutable).
  - No hardcoded magic numbers; all thresholds are named constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

_MIN_BARS: int = 3          # need at least 3 bars to form one FVG
_MIN_GAP_PCT: float = 0.0   # accept any non-zero gap by default


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FairValueGap:
    """A single Fair Value Gap (price imbalance)."""

    kind: str        # "bullish_fvg" or "bearish_fvg"
    upper: float     # top boundary of the gap
    lower: float     # bottom boundary of the gap
    size_pct: float  # gap size as a percentage of the midpoint price
    index: int       # bar index of the middle (driving) candle
    filled: bool     # True if subsequent price has re-entered the gap


# ── Detector ──────────────────────────────────────────────────────────────────

class FVGDetector:
    """
    Detects Fair Value Gaps from an OHLCV DataFrame.

    Parameters
    ----------
    min_gap_pct : float
        Minimum gap size as a percentage of price to qualify.
        Set to 0.0 (default) to capture all gaps.
    """

    def __init__(self, min_gap_pct: float = _MIN_GAP_PCT) -> None:
        self._min_gap_pct = min_gap_pct

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> list[FairValueGap]:
        """
        Scan *df* for Fair Value Gaps and return a list of ``FairValueGap`` objects.

        Returns an empty list for empty or too-short DataFrames without raising.
        """
        if df is None or df.empty or len(df) < _MIN_BARS:
            return []

        required = {"high", "low", "close"}
        if not required.issubset(df.columns):
            return []

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)

        n = len(highs)
        fvgs: list[FairValueGap] = []

        # Each FVG is centred on bar i, using bars i-1 and i+1
        for i in range(1, n - 1):
            gap_low, gap_high, kind = _extract_gap(highs, lows, i)

            if gap_low is None or gap_high is None or kind is None:
                continue

            # Ensure a positive gap exists
            gap_size = gap_high - gap_low
            if gap_size <= 0:
                continue

            mid_price = (gap_high + gap_low) / 2.0
            size_pct = (gap_size / mid_price * 100.0) if mid_price > 0 else 0.0

            if size_pct < self._min_gap_pct:
                continue

            # Check if price has returned to fill the gap (bars after i+1)
            filled = _is_filled(highs, lows, i + 2, gap_low, gap_high, kind)

            fvgs.append(
                FairValueGap(
                    kind=kind,
                    upper=float(gap_high),
                    lower=float(gap_low),
                    size_pct=round(size_pct, 4),
                    index=i,
                    filled=filled,
                )
            )

        return fvgs


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_gap(
    highs: np.ndarray,
    lows: np.ndarray,
    i: int,
) -> tuple[float | None, float | None, str | None]:
    """
    Determine whether bar *i* is the middle candle of an FVG.

    Returns (gap_low, gap_high, kind) or (None, None, None) if no gap.
    """
    prev_high = float(highs[i - 1])
    prev_low = float(lows[i - 1])
    next_high = float(highs[i + 1])
    next_low = float(lows[i + 1])

    # Bullish FVG: previous candle's high < next candle's low
    if prev_high < next_low:
        return prev_high, next_low, "bullish_fvg"

    # Bearish FVG: previous candle's low > next candle's high
    if prev_low > next_high:
        return next_high, prev_low, "bearish_fvg"

    return None, None, None


def _is_filled(
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    gap_low: float,
    gap_high: float,
    kind: str,
) -> bool:
    """
    Return True if price has re-entered the gap in bars from *start* onward.

    For a bullish FVG (gap = buying imbalance above), it is filled when
    a subsequent bar's low trades into or below the gap's upper boundary.
    For a bearish FVG (gap = selling imbalance below), filled when a bar's
    high trades into or above the gap's lower boundary.
    """
    n = len(highs)
    for j in range(start, n):
        bar_high = float(highs[j])
        bar_low = float(lows[j])
        # Any overlap with the gap zone counts as filled
        if bar_low <= gap_high and bar_high >= gap_low:
            return True
    return False

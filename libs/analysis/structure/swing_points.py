"""
Swing Point Detector.

Detects swing highs and swing lows from OHLCV data and classifies each
pivot as HH (Higher High), HL (Higher Low), LH (Lower High), or LL (Lower Low).

Design rules:
  - Never raises on empty or short DataFrames — returns an empty list.
  - All output objects are frozen dataclasses (immutable).
  - No magic numbers; all thresholds are named constants or constructor params.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# Number of recent swing points considered by classify_trend.
_TREND_WINDOW: int = 6

# Minimum counts required to declare a direction in classify_trend.
_MIN_TREND_COUNT: int = 1


@dataclass(frozen=True)
class SwingPoint:
    """A single identified swing pivot with its HH/HL/LH/LL classification."""

    index: int           # bar index in the DataFrame
    price: float         # high value for swing high, low value for swing low
    kind: str            # "high" or "low"
    classification: str  # "HH", "HL", "LH", "LL", or "first"


class SwingPointDetector:
    """
    Detects swing highs and swing lows from an OHLCV DataFrame.

    Parameters
    ----------
    lookback : int
        Number of bars on each side a bar must be the extreme value of
        to qualify as a swing point.  Larger values → fewer, more significant
        pivots.  Default is 2.
    """

    def __init__(self, lookback: int = 2) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> list[SwingPoint]:
        """
        Detect and classify all swing highs and lows in `df`.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with at least "high" and "low" columns.

        Returns
        -------
        list[SwingPoint]
            All detected swing points sorted by bar index.  Returns [] for
            empty or too-short DataFrames.
        """
        if df is None or df.empty:
            return []

        n = len(df)
        # Need at least (lookback + 1 + lookback) bars to find any pivot.
        if n < 2 * self.lookback + 1:
            return []

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()

        swing_high_indices = self._find_swing_highs(highs)
        swing_low_indices = self._find_swing_lows(lows)

        classified_highs = self._classify_highs(swing_high_indices, highs)
        classified_lows = self._classify_lows(swing_low_indices, lows)

        all_points = classified_highs + classified_lows
        all_points.sort(key=lambda p: p.index)
        return all_points

    # ── Pivot detection ───────────────────────────────────────────────────────

    def _find_swing_highs(self, highs: "np.ndarray") -> list[int]:  # type: ignore[name-defined]  # noqa: F821
        """
        Return sorted list of bar indices that qualify as swing highs.

        Bar at index i is a swing high if:
            high[i] > high[i-j]  AND  high[i] > high[i+j]
            for all j in 1..lookback
        """
        n = len(highs)
        result: list[int] = []
        for i in range(self.lookback, n - self.lookback):
            candidate = highs[i]
            is_high = all(
                candidate > highs[i - j] and candidate > highs[i + j]
                for j in range(1, self.lookback + 1)
            )
            if is_high:
                result.append(i)
        return result

    def _find_swing_lows(self, lows: "np.ndarray") -> list[int]:  # type: ignore[name-defined]  # noqa: F821
        """
        Return sorted list of bar indices that qualify as swing lows.

        Bar at index i is a swing low if:
            low[i] < low[i-j]  AND  low[i] < low[i+j]
            for all j in 1..lookback
        """
        n = len(lows)
        result: list[int] = []
        for i in range(self.lookback, n - self.lookback):
            candidate = lows[i]
            is_low = all(
                candidate < lows[i - j] and candidate < lows[i + j]
                for j in range(1, self.lookback + 1)
            )
            if is_low:
                result.append(i)
        return result

    # ── Classification ────────────────────────────────────────────────────────

    @staticmethod
    def _classify_highs(
        indices: list[int],
        highs: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    ) -> list[SwingPoint]:
        """
        Classify a sequence of swing high indices as "first", "HH", or "LH".

        The first swing high always gets classification="first".
        Subsequent highs:
          - "HH" if price > previous swing high price
          - "LH" otherwise
        """
        points: list[SwingPoint] = []
        prev_price: float | None = None

        for idx in indices:
            price = float(highs[idx])
            if prev_price is None:
                classification = "first"
            elif price > prev_price:
                classification = "HH"
            else:
                classification = "LH"

            points.append(
                SwingPoint(index=idx, price=price, kind="high", classification=classification)
            )
            prev_price = price

        return points

    @staticmethod
    def _classify_lows(
        indices: list[int],
        lows: "np.ndarray",  # type: ignore[name-defined]  # noqa: F821
    ) -> list[SwingPoint]:
        """
        Classify a sequence of swing low indices as "first", "HL", or "LL".

        The first swing low always gets classification="first".
        Subsequent lows:
          - "HL" if price > previous swing low price  (higher low)
          - "LL" otherwise
        """
        points: list[SwingPoint] = []
        prev_price: float | None = None

        for idx in indices:
            price = float(lows[idx])
            if prev_price is None:
                classification = "first"
            elif price > prev_price:
                classification = "HL"
            else:
                classification = "LL"

            points.append(
                SwingPoint(index=idx, price=price, kind="low", classification=classification)
            )
            prev_price = price

        return points


# ── Standalone trend classifier ───────────────────────────────────────────────

def classify_trend(points: list[SwingPoint]) -> str:
    """
    Classify the recent market trend from a list of swing points.

    Examines the last ``_TREND_WINDOW`` (6) swing points, counts the HH, HL,
    LH, and LL labels (excluding "first"), and returns:

    - ``"uptrend"``   if HH+HL > LH+LL  AND  HH >= 1  AND  HL >= 1
    - ``"downtrend"`` if LH+LL > HH+HL  AND  LH >= 1  AND  LL >= 1
    - ``"neutral"``   otherwise

    Parameters
    ----------
    points : list[SwingPoint]
        All detected swing points (mixed highs and lows), as returned by
        :meth:`SwingPointDetector.detect`.

    Returns
    -------
    str
        One of ``"uptrend"``, ``"downtrend"``, or ``"neutral"``.
    """
    # Consider only the most recent _TREND_WINDOW points.
    recent = points[-_TREND_WINDOW:]

    counts: dict[str, int] = {"HH": 0, "HL": 0, "LH": 0, "LL": 0}
    for pt in recent:
        if pt.classification in counts:
            counts[pt.classification] += 1

    bullish = counts["HH"] + counts["HL"]
    bearish = counts["LH"] + counts["LL"]

    if (
        bullish > bearish
        and counts["HH"] >= _MIN_TREND_COUNT
        and counts["HL"] >= _MIN_TREND_COUNT
    ):
        return "uptrend"

    if (
        bearish > bullish
        and counts["LH"] >= _MIN_TREND_COUNT
        and counts["LL"] >= _MIN_TREND_COUNT
    ):
        return "downtrend"

    return "neutral"

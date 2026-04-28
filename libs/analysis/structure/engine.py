"""
Market Structure Engine.

Identifies swing highs / lows, classifies trend direction (HH/HL, LH/LL),
measures trend strength, and detects Breaks of Structure (BoS).

Design rules:
  - Never raises on empty or short DataFrames — returns safe UNKNOWN defaults.
  - All output objects are frozen dataclasses (immutable).
  - No magic numbers; all thresholds are named constants or constructor params.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from libs.core.models.domain import TrendDirection

# Minimum number of total pivots (highs + lows combined) required
# before we attempt trend classification.
_MIN_PIVOTS_FOR_CLASSIFICATION: int = 4

# Minimum pivot highs and lows needed to classify (2 each).
_MIN_PIVOTS_EACH: int = 2


# ── Output models ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructurePoint:
    """A single identified swing pivot with its classification label."""

    idx: int
    price: float
    point_type: str          # "HH", "HL", "LH", "LL", "pivot_high", "pivot_low"
    timestamp: datetime | None = None


@dataclass(frozen=True)
class MarketStructure:
    """
    Complete market structure snapshot for the analysed window.

    trend_strength is 0.0–1.0: proportion of consecutive pivot pairs
    that conform to the detected trend direction.
    """

    trend: TrendDirection
    points: list[StructurePoint]
    trend_strength: float        # 0.0 – 1.0
    break_of_structure: bool     # True if latest close broke prior swing against trend
    swing_high: float | None     # most recent confirmed swing high
    swing_low: float | None      # most recent confirmed swing low
    notes: str = ""


# ── Engine ────────────────────────────────────────────────────────────────────

class MarketStructureEngine:
    """
    Detects market structure from an OHLCV DataFrame.

    Parameters
    ----------
    pivot_strength : int
        Number of bars on each side a bar must be the highest/lowest to
        qualify as a pivot.  Larger values → fewer, more significant pivots.
    lookback : int
        Only the last `lookback` rows of the DataFrame are considered.
    """

    def __init__(
        self,
        pivot_strength: int = 5,
        lookback: int = 100,
    ) -> None:
        self.pivot_strength = pivot_strength
        self.lookback = lookback

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> MarketStructure:
        """
        Analyse the DataFrame and return a MarketStructure.

        Never raises — returns UNKNOWN trend on empty / insufficient data.
        """
        if df is None or df.empty:
            return self._unknown("Empty DataFrame")

        # Work on the lookback window only
        window = df.tail(self.lookback).copy()

        if len(window) < (2 * self.pivot_strength + 1):
            return self._unknown(
                f"Insufficient bars ({len(window)}) for pivot_strength={self.pivot_strength}"
            )

        high_series = window["high"]
        low_series = window["low"]

        # Find raw pivot indices (relative to `window`)
        ph_indices = self._find_pivot_highs(high_series, self.pivot_strength)
        pl_indices = self._find_pivot_lows(low_series, self.pivot_strength)

        if len(ph_indices) + len(pl_indices) < _MIN_PIVOTS_FOR_CLASSIFICATION:
            return self._unknown(
                f"Too few pivots found (highs={len(ph_indices)}, lows={len(pl_indices)})"
            )

        # Build (index, price) tuples
        highs: list[tuple[int, float]] = [
            (i, float(high_series.iloc[i])) for i in ph_indices
        ]
        lows: list[tuple[int, float]] = [
            (i, float(low_series.iloc[i])) for i in pl_indices
        ]

        trend, structure_points, strength = self._classify_pivots(highs, lows)

        # Attach timestamps where available
        if hasattr(window.index, "__iter__"):
            idx_arr = list(window.index)
            enriched_points = [
                StructurePoint(
                    idx=sp.idx,
                    price=sp.price,
                    point_type=sp.point_type,
                    timestamp=idx_arr[sp.idx] if sp.idx < len(idx_arr) else None,
                )
                for sp in structure_points
            ]
        else:
            enriched_points = structure_points

        # Determine swing high / swing low from raw pivots
        swing_high: float | None = highs[-1][1] if highs else None
        swing_low: float | None = lows[-1][1] if lows else None

        bos = self._detect_bos(window, swing_high, swing_low, trend)

        return MarketStructure(
            trend=trend,
            points=enriched_points,
            trend_strength=round(strength, 4),
            break_of_structure=bos,
            swing_high=swing_high,
            swing_low=swing_low,
        )

    # ── Pivot detection ───────────────────────────────────────────────────────

    def _find_pivot_highs(self, series: pd.Series, strength: int) -> list[int]:
        """
        Return list of integer positions (iloc-based) that are pivot highs.

        A bar at position i is a pivot high if:
            series.iloc[i] == max(series.iloc[i-strength : i+strength+1])
        """
        values = series.to_numpy()
        n = len(values)
        pivots: list[int] = []
        for i in range(strength, n - strength):
            window_max = values[i - strength: i + strength + 1].max()
            if values[i] == window_max:
                pivots.append(i)
        return pivots

    def _find_pivot_lows(self, series: pd.Series, strength: int) -> list[int]:
        """
        Return list of integer positions (iloc-based) that are pivot lows.

        A bar at position i is a pivot low if:
            series.iloc[i] == min(series.iloc[i-strength : i+strength+1])
        """
        values = series.to_numpy()
        n = len(values)
        pivots: list[int] = []
        for i in range(strength, n - strength):
            window_min = values[i - strength: i + strength + 1].min()
            if values[i] == window_min:
                pivots.append(i)
        return pivots

    # ── Trend classification ──────────────────────────────────────────────────

    def _classify_pivots(
        self,
        highs: list[tuple[int, float]],
        lows: list[tuple[int, float]],
    ) -> tuple[TrendDirection, list[StructurePoint], float]:
        """
        Classify pivot highs and lows into HH/HL/LH/LL labels and determine
        the overall trend direction.

        Returns
        -------
        (trend, structure_points, trend_strength)
        """
        if len(highs) < _MIN_PIVOTS_EACH or len(lows) < _MIN_PIVOTS_EACH:
            raw_points = [
                StructurePoint(idx=i, price=p, point_type="pivot_high")
                for i, p in highs
            ] + [
                StructurePoint(idx=i, price=p, point_type="pivot_low")
                for i, p in lows
            ]
            return TrendDirection.UNKNOWN, raw_points, 0.0

        # Label each high relative to the previous high
        labeled_highs: list[StructurePoint] = []
        for k in range(len(highs)):
            i, price = highs[k]
            if k == 0:
                label = "pivot_high"
            else:
                prev_price = highs[k - 1][1]
                label = "HH" if price > prev_price else "LH"
            labeled_highs.append(StructurePoint(idx=i, price=price, point_type=label))

        # Label each low relative to the previous low
        labeled_lows: list[StructurePoint] = []
        for k in range(len(lows)):
            i, price = lows[k]
            if k == 0:
                label = "pivot_low"
            else:
                prev_price = lows[k - 1][1]
                label = "HL" if price > prev_price else "LL"
            labeled_lows.append(StructurePoint(idx=i, price=price, point_type=label))

        # Determine trend from the two most recent labels of each type
        last_high_label = labeled_highs[-1].point_type   # HH or LH
        prev_high_label = labeled_highs[-2].point_type   # HH or LH
        last_low_label = labeled_lows[-1].point_type     # HL or LL
        prev_low_label = labeled_lows[-2].point_type     # HL or LL

        uptrend_highs = {"HH"}
        uptrend_lows = {"HL"}
        downtrend_highs = {"LH"}
        downtrend_lows = {"LL"}

        is_uptrend = (
            last_high_label in uptrend_highs
            and prev_high_label in uptrend_highs
            and last_low_label in uptrend_lows
            and prev_low_label in uptrend_lows
        )
        is_downtrend = (
            last_high_label in downtrend_highs
            and prev_high_label in downtrend_highs
            and last_low_label in downtrend_lows
            and prev_low_label in downtrend_lows
        )

        if is_uptrend:
            trend = TrendDirection.UPTREND
        elif is_downtrend:
            trend = TrendDirection.DOWNTREND
        else:
            trend = TrendDirection.RANGING

        # Compute trend_strength: proportion of consecutive pivot pairs
        # (both highs and lows) that conform to the detected trend.
        strength = self._compute_trend_strength(
            labeled_highs, labeled_lows, trend
        )

        all_points = sorted(
            labeled_highs + labeled_lows, key=lambda sp: sp.idx
        )

        return trend, all_points, strength

    def _compute_trend_strength(
        self,
        labeled_highs: list[StructurePoint],
        labeled_lows: list[StructurePoint],
        trend: TrendDirection,
    ) -> float:
        """
        Ratio of conforming consecutive pairs to total pairs examined.

        For UPTREND: conforming high pair = HH→HH, conforming low pair = HL→HL.
        For DOWNTREND: LH→LH and LL→LL.
        For RANGING / UNKNOWN: 0.0.
        """
        if trend not in (TrendDirection.UPTREND, TrendDirection.DOWNTREND):
            return 0.0

        conforming_label_high = "HH" if trend == TrendDirection.UPTREND else "LH"
        conforming_label_low = "HL" if trend == TrendDirection.UPTREND else "LL"

        total_pairs = 0
        conforming_pairs = 0

        # Consecutive high pairs (skip the first "pivot_high" label)
        classifiable_highs = [sp for sp in labeled_highs if sp.point_type in ("HH", "LH")]
        for k in range(1, len(classifiable_highs)):
            total_pairs += 1
            if (
                classifiable_highs[k - 1].point_type == conforming_label_high
                and classifiable_highs[k].point_type == conforming_label_high
            ):
                conforming_pairs += 1

        # Consecutive low pairs
        classifiable_lows = [sp for sp in labeled_lows if sp.point_type in ("HL", "LL")]
        for k in range(1, len(classifiable_lows)):
            total_pairs += 1
            if (
                classifiable_lows[k - 1].point_type == conforming_label_low
                and classifiable_lows[k].point_type == conforming_label_low
            ):
                conforming_pairs += 1

        if total_pairs == 0:
            return 0.0

        return conforming_pairs / total_pairs

    # ── Break of Structure ────────────────────────────────────────────────────

    def _detect_bos(
        self,
        df: pd.DataFrame,
        swing_high: float | None,
        swing_low: float | None,
        trend: TrendDirection,
    ) -> bool:
        """
        Return True if the latest close has broken the prior swing level
        against the current trend.

        UPTREND   → BoS if latest close < swing_low
        DOWNTREND → BoS if latest close > swing_high
        RANGING / UNKNOWN → False
        """
        if df.empty:
            return False

        latest_close = float(df["close"].iloc[-1])

        if trend == TrendDirection.UPTREND and swing_low is not None:
            return latest_close < swing_low

        if trend == TrendDirection.DOWNTREND and swing_high is not None:
            return latest_close > swing_high

        return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _unknown(notes: str = "") -> MarketStructure:
        """Return a safe UNKNOWN MarketStructure."""
        return MarketStructure(
            trend=TrendDirection.UNKNOWN,
            points=[],
            trend_strength=0.0,
            break_of_structure=False,
            swing_high=None,
            swing_low=None,
            notes=notes,
        )

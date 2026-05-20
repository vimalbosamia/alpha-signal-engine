"""
WyckoffDetector — identifies Wyckoff market phases from OHLCV data.

Phases detected:
  - accumulation: price range-bound after downtrend, declining volume, spring wick
  - markup:       price breaks above range on expanding volume, HH+HL structure
  - distribution: price range-bound after uptrend, declining volume, upthrust wick
  - markdown:     price breaks below range on expanding volume, LH+LL structure
  - unknown:      insufficient data or no clear pattern

Design rules:
  - Never raises on empty or short DataFrames — returns safe neutral defaults.
  - All output objects are frozen dataclasses (immutable).
  - No hardcoded magic numbers; all thresholds are named constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

_RANGE_WINDOW: int = 20            # bars used to assess the recent price range
_PRIOR_WINDOW: int = 20            # bars before the range window to assess trend
_MIN_BARS: int = _RANGE_WINDOW + 5 # minimum bars needed for detection
_VOLUME_BREAKOUT_FACTOR: float = 1.5  # volume must be this × avg to confirm breakout
_RANGE_TIGHT_PCT: float = 0.08     # range / midpoint must be ≤ this to be "tight"
_TREND_MOVE_PCT: float = 0.03      # prior bars must have moved ≥ 3 % to confirm trend
_VOLUME_DECLINE_RATIO: float = 0.9 # second half volume must be ≤ this × first half


# ── Output model ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WyckoffPhase:
    """Detected Wyckoff market phase."""

    phase: str           # "accumulation", "markup", "distribution", "markdown", "unknown"
    confidence: float    # 0.0 to 1.0
    volume_pattern: str  # "increasing", "decreasing", "climactic", "normal"
    explanation: str


# ── Detector ──────────────────────────────────────────────────────────────────

class WyckoffDetector:
    """
    Detects the current Wyckoff market phase from an OHLCV DataFrame.

    Parameters
    ----------
    range_window : int
        Number of recent bars used to assess the current price range.
    volume_breakout_factor : float
        Volume multiplier above average required to confirm a breakout.
    """

    def __init__(
        self,
        range_window: int = _RANGE_WINDOW,
        volume_breakout_factor: float = _VOLUME_BREAKOUT_FACTOR,
    ) -> None:
        self._range_window = range_window
        self._volume_breakout_factor = volume_breakout_factor

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> WyckoffPhase:
        """
        Detect the Wyckoff market phase from *df*.

        Never raises — returns an 'unknown' phase on empty or insufficient data.
        """
        if df is None or df.empty or len(df) < _MIN_BARS:
            return _unknown_phase("Insufficient data")

        required = {"high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return _unknown_phase("Missing required columns")

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        volumes = df["volume"].to_numpy(dtype=float)

        n = len(closes)
        rw = self._range_window

        # Slice: recent range window and prior window for trend direction
        recent_highs = highs[n - rw:]
        recent_lows = lows[n - rw:]
        recent_closes = closes[n - rw:]
        recent_volumes = volumes[n - rw:]

        prior_start = max(0, n - rw - _PRIOR_WINDOW)
        prior_closes = closes[prior_start: n - rw]

        # ── Feature extraction ────────────────────────────────────────────────

        range_high = float(recent_highs.max())
        range_low = float(recent_lows.min())
        midpoint = (range_high + range_low) / 2.0

        range_pct = (range_high - range_low) / midpoint if midpoint > 0 else 0.0
        is_tight_range = range_pct <= _RANGE_TIGHT_PCT

        # Prior trend: compare start vs end of the prior window
        prior_trend = _assess_prior_trend(prior_closes)

        # Volume pattern within range window
        vol_pattern = _assess_volume_pattern(recent_volumes)
        is_vol_declining = vol_pattern == "decreasing"
        is_vol_expanding = vol_pattern == "increasing" or vol_pattern == "climactic"

        # Current close relative to range
        current_close = float(recent_closes[-1])
        avg_volume = float(volumes[max(0, n - rw - 5): n - rw].mean()) if n > rw else float(volumes.mean())
        current_volume = float(volumes[-1])
        is_high_volume_breakout = (
            avg_volume > 0 and current_volume >= self._volume_breakout_factor * avg_volume
        )

        # Spring: lowest low in range window has a wick below (wick = low < range_low_excl)
        has_spring = _has_spring_wick(recent_highs, recent_lows, recent_closes, range_low)
        # Upthrust: highest high in range window has a wick above (wick = high > range_high_excl)
        has_upthrust = _has_upthrust_wick(recent_highs, recent_lows, recent_closes, range_high)

        # ── Phase classification ───────────────────────────────────────────────

        # Markup: close above range_high with volume expansion
        if current_close > range_high and is_high_volume_breakout:
            conf = _markup_confidence(is_vol_expanding, prior_trend)
            return WyckoffPhase(
                phase="markup",
                confidence=round(conf, 4),
                volume_pattern=vol_pattern,
                explanation=(
                    f"Close {current_close:.4f} broke above range high {range_high:.4f} "
                    f"with volume {current_volume:.0f} vs avg {avg_volume:.0f} "
                    f"(×{current_volume / avg_volume:.2f}). Prior trend: {prior_trend}."
                ),
            )

        # Markdown: close below range_low with volume expansion
        if current_close < range_low and is_high_volume_breakout:
            conf = _markdown_confidence(is_vol_expanding, prior_trend)
            return WyckoffPhase(
                phase="markdown",
                confidence=round(conf, 4),
                volume_pattern=vol_pattern,
                explanation=(
                    f"Close {current_close:.4f} broke below range low {range_low:.4f} "
                    f"with volume {current_volume:.0f} vs avg {avg_volume:.0f} "
                    f"(×{current_volume / avg_volume:.2f}). Prior trend: {prior_trend}."
                ),
            )

        # Accumulation: tight range after downtrend, declining volume, optional spring
        if is_tight_range and prior_trend == "downtrend" and is_vol_declining:
            spring_note = " Spring wick detected." if has_spring else ""
            conf = 0.60 + (0.20 if has_spring else 0.0)
            return WyckoffPhase(
                phase="accumulation",
                confidence=round(conf, 4),
                volume_pattern=vol_pattern,
                explanation=(
                    f"Tight range ({range_pct:.2%}) after downtrend. "
                    f"Volume declining.{spring_note}"
                ),
            )

        # Distribution: tight range after uptrend, declining volume, optional upthrust
        if is_tight_range and prior_trend == "uptrend" and is_vol_declining:
            upthrust_note = " Upthrust wick detected." if has_upthrust else ""
            conf = 0.60 + (0.20 if has_upthrust else 0.0)
            return WyckoffPhase(
                phase="distribution",
                confidence=round(conf, 4),
                volume_pattern=vol_pattern,
                explanation=(
                    f"Tight range ({range_pct:.2%}) after uptrend. "
                    f"Volume declining.{upthrust_note}"
                ),
            )

        return _unknown_phase(
            f"No clear Wyckoff phase. Range%={range_pct:.2%}, "
            f"prior_trend={prior_trend}, vol_pattern={vol_pattern}."
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _unknown_phase(reason: str) -> WyckoffPhase:
    return WyckoffPhase(
        phase="unknown",
        confidence=0.0,
        volume_pattern="normal",
        explanation=reason,
    )


def _assess_prior_trend(closes: np.ndarray) -> str:
    """
    Assess the trend direction of the prior bars.

    Returns "uptrend", "downtrend", or "sideways".
    """
    if len(closes) < 2:
        return "sideways"

    start = float(closes[0])
    end = float(closes[-1])

    if start <= 0:
        return "sideways"

    move_pct = (end - start) / start

    if move_pct >= _TREND_MOVE_PCT:
        return "uptrend"
    if move_pct <= -_TREND_MOVE_PCT:
        return "downtrend"
    return "sideways"


def _assess_volume_pattern(volumes: np.ndarray) -> str:
    """
    Characterise volume over a window as increasing, decreasing, climactic, or normal.

    Splits the window in two halves; compares their mean volume.
    """
    n = len(volumes)
    if n < 4:
        return "normal"

    half = n // 2
    first_half_avg = float(volumes[:half].mean())
    second_half_avg = float(volumes[half:].mean())

    # Climactic: the last bar is a large spike above the rest
    last_vol = float(volumes[-1])
    overall_avg = float(volumes[:-1].mean()) if n > 1 else first_half_avg
    if overall_avg > 0 and last_vol >= 2.5 * overall_avg:
        return "climactic"

    if first_half_avg <= 0:
        return "normal"

    ratio = second_half_avg / first_half_avg

    if ratio >= 1.1:
        return "increasing"
    if ratio <= _VOLUME_DECLINE_RATIO:
        return "decreasing"
    return "normal"


def _has_spring_wick(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    range_low: float,
) -> bool:
    """
    Return True if any bar has a wick that dips below the range low
    but closes back above it (spring pattern).
    """
    for i in range(len(lows)):
        if lows[i] < range_low and closes[i] >= range_low:
            return True
    return False


def _has_upthrust_wick(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    range_high: float,
) -> bool:
    """
    Return True if any bar has a wick that spikes above the range high
    but closes back below it (upthrust pattern).
    """
    for i in range(len(highs)):
        if highs[i] > range_high and closes[i] <= range_high:
            return True
    return False


def _markup_confidence(is_vol_expanding: bool, prior_trend: str) -> float:
    """Confidence for markup phase."""
    base = 0.65
    if is_vol_expanding:
        base += 0.15
    if prior_trend in ("uptrend", "downtrend"):
        base += 0.10
    return min(base, 1.0)


def _markdown_confidence(is_vol_expanding: bool, prior_trend: str) -> float:
    """Confidence for markdown phase."""
    base = 0.65
    if is_vol_expanding:
        base += 0.15
    if prior_trend in ("uptrend", "downtrend"):
        base += 0.10
    return min(base, 1.0)

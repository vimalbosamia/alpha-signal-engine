"""
MarketStructureAnalyzer — detects Break of Structure (BOS) and Change of
Character (CHoCH) events from swing points and price action.

Design rules:
  - Never raises on empty or short DataFrames — returns safe neutral defaults.
  - All output objects are frozen dataclasses (immutable).
  - No magic numbers; all thresholds are named constants or constructor params.

BOS (Break of Structure):
  - Bullish BOS: a closing price breaks above a previous swing high.
  - Bearish BOS: a closing price breaks below a previous swing low.

CHoCH (Change of Character):
  - A BOS that contradicts the prior trend bias.
  - E.g., in a downtrend (LH+LL) a bullish BOS above a swing high is a CHoCH.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    from libs.analysis.structure.swing_points import (
        SwingPoint,
        SwingPointDetector,
        classify_trend,
    )
except ImportError:  # pragma: no cover — stub provided in tests
    raise


# ── Output models ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StructureEvent:
    """A single BOS or CHoCH event detected from price action."""

    kind: str           # "BOS" or "CHoCH"
    direction: str      # "bullish" or "bearish"
    index: int          # bar index (iloc position) where break occurred
    price: float        # closing price that broke the structure
    broken_level: float  # the swing level that was broken
    explanation: str


@dataclass(frozen=True)
class StructureAnalysis:
    """Complete market structure snapshot for the analysed window."""

    trend_bias: str                   # "bullish", "bearish", or "neutral"
    swing_points: list[SwingPoint]
    events: list[StructureEvent]
    strength: float                   # 0.0 to 1.0
    explanation: str
    last_swing_high: float | None
    last_swing_low: float | None


# ── Constants ──────────────────────────────────────────────────────────────────

_NEUTRAL = "neutral"
_BULLISH = "bullish"
_BEARISH = "bearish"

# classify_trend() from swing_points returns these values
_UPTREND = "uptrend"
_DOWNTREND = "downtrend"

# Minimum number of swing points needed before BOS/CHoCH detection is attempted.
_MIN_SWING_POINTS: int = 2


def _normalize_trend(raw: str) -> str:
    """Normalise classify_trend output to bullish/bearish/neutral."""
    if raw == _UPTREND:
        return _BULLISH
    if raw == _DOWNTREND:
        return _BEARISH
    return _NEUTRAL


# ── Analyzer ───────────────────────────────────────────────────────────────────

class MarketStructureAnalyzer:
    """
    Detects market structure events (BOS, CHoCH) from an OHLCV DataFrame.

    Parameters
    ----------
    lookback : int
        Passed to the underlying SwingPointDetector as the number of bars on
        each side a bar must be the local extreme to qualify as a swing point.
    """

    def __init__(self, lookback: int = 2) -> None:
        self._detector = SwingPointDetector(lookback=lookback)

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> StructureAnalysis:
        """
        Analyse the DataFrame and return a StructureAnalysis.

        Never raises — returns neutral defaults on empty / insufficient data.
        """
        if df is None or df.empty:
            return _neutral_result("Empty DataFrame")

        swing_points = self._detector.detect(df)

        if len(swing_points) < _MIN_SWING_POINTS:
            return _neutral_result(
                f"Insufficient swing points ({len(swing_points)})"
            )

        # Compute prior trend from the "historical" portion of swing points
        # (first 75%, minimum 2) so that a recent breakout does not distort the
        # baseline trend used for CHoCH classification.
        prior_trend = _normalize_trend(classify_trend(_historical_points(swing_points)))

        last_swing_high, last_swing_low = _extract_last_swings(swing_points)

        events = _detect_events(df, swing_points, prior_trend)

        trend_bias = _determine_trend_bias(events, prior_trend)
        strength = _compute_strength(swing_points, events)
        explanation = _build_explanation(trend_bias, events, swing_points)

        return StructureAnalysis(
            trend_bias=trend_bias,
            swing_points=swing_points,
            events=events,
            strength=round(strength, 4),
            explanation=explanation,
            last_swing_high=last_swing_high,
            last_swing_low=last_swing_low,
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _historical_points(points: list[SwingPoint]) -> list[SwingPoint]:
    """
    Return the "historical" portion of swing points for prior-trend classification.

    Uses the first 75% of points (minimum 2) so that a recent breakout surge
    does not pollute the baseline trend used for CHoCH detection.
    """
    n = len(points)
    if n <= 2:
        return points
    cutoff = max(2, int(n * 0.75))
    return points[:cutoff]


def _neutral_result(reason: str) -> StructureAnalysis:
    """Return a safe neutral StructureAnalysis."""
    return StructureAnalysis(
        trend_bias=_NEUTRAL,
        swing_points=[],
        events=[],
        strength=0.0,
        explanation=reason,
        last_swing_high=None,
        last_swing_low=None,
    )


def _extract_last_swings(
    swing_points: list[SwingPoint],
) -> tuple[float | None, float | None]:
    """Return the last detected swing high and swing low prices."""
    highs = [p for p in swing_points if p.kind == "high"]
    lows = [p for p in swing_points if p.kind == "low"]
    last_high = highs[-1].price if highs else None
    last_low = lows[-1].price if lows else None
    return last_high, last_low


def _detect_events(
    df: pd.DataFrame,
    swing_points: list[SwingPoint],
    prior_trend: str,
) -> list[StructureEvent]:
    """
    Scan every closing price against previously confirmed swing levels.

    For each bar (close), check:
      - If close > any previously confirmed swing high → bullish BOS (or CHoCH)
      - If close < any previously confirmed swing low  → bearish BOS (or CHoCH)

    A CHoCH is a BOS that contradicts the prior trend:
      - Bullish BOS in a bearish/downtrend → CHoCH
      - Bearish BOS in a bullish/uptrend   → CHoCH
    """
    closes = df["close"].to_numpy(dtype=float)
    n = len(closes)
    events: list[StructureEvent] = []

    # Build maps of confirmed swing levels indexed by bar position
    confirmed_highs: list[SwingPoint] = []
    confirmed_lows: list[SwingPoint] = []

    # Running trend context starts from classify_trend result;
    # it updates as CHoCH events are emitted.
    running_trend = prior_trend

    sp_by_index: dict[int, list[SwingPoint]] = {}
    for sp in swing_points:
        sp_by_index.setdefault(sp.index, []).append(sp)

    for bar_idx in range(n):
        # First, check for breaks using currently confirmed levels
        close = closes[bar_idx]

        # Check bullish BOS: close > confirmed swing high
        for sh in confirmed_highs:
            if close > sh.price:
                is_choch = running_trend == _BEARISH
                kind = "CHoCH" if is_choch else "BOS"
                expl = (
                    f"Close {close:.4f} broke above swing high {sh.price:.4f} "
                    f"at bar {sh.index} → {'CHoCH' if is_choch else 'Bullish BOS'}"
                )
                events.append(
                    StructureEvent(
                        kind=kind,
                        direction=_BULLISH,
                        index=bar_idx,
                        price=float(close),
                        broken_level=float(sh.price),
                        explanation=expl,
                    )
                )
                # Update running trend context on CHoCH
                if is_choch:
                    running_trend = _BULLISH

        # Check bearish BOS: close < confirmed swing low
        for sl in confirmed_lows:
            if close < sl.price:
                is_choch = running_trend == _BULLISH
                kind = "CHoCH" if is_choch else "BOS"
                expl = (
                    f"Close {close:.4f} broke below swing low {sl.price:.4f} "
                    f"at bar {sl.index} → {'CHoCH' if is_choch else 'Bearish BOS'}"
                )
                events.append(
                    StructureEvent(
                        kind=kind,
                        direction=_BEARISH,
                        index=bar_idx,
                        price=float(close),
                        broken_level=float(sl.price),
                        explanation=expl,
                    )
                )
                if is_choch:
                    running_trend = _BEARISH

        # After checking, confirm any swing points AT this bar index
        if bar_idx in sp_by_index:
            for sp in sp_by_index[bar_idx]:
                if sp.kind == "high":
                    confirmed_highs.append(sp)
                else:
                    confirmed_lows.append(sp)

    return events


def _determine_trend_bias(
    events: list[StructureEvent],
    prior_trend: str,
) -> str:
    """
    Determine the final trend bias:
      1. Direction of the last BOS or CHoCH event (if any).
      2. Falls back to prior_trend from classify_trend.
    """
    if events:
        return events[-1].direction
    return prior_trend


def _compute_strength(
    swing_points: list[SwingPoint],
    events: list[StructureEvent],
) -> float:
    """
    Compute structure strength as a value in [0.0, 1.0].

    Components:
      - Pattern consistency: proportion of swing-point labels that conform to
        the dominant trend (HH/HL for bullish, LH/LL for bearish).
      - Event weight: bonus for each BOS (+0.05) and CHoCH (+0.10), capped.
    """
    # "first" is the initial classification used by SwingPointDetector;
    # only consider labelled (non-first) points for strength computation.
    highs = [p for p in swing_points if p.kind == "high" and p.classification in ("HH", "LH")]
    lows = [p for p in swing_points if p.kind == "low" and p.classification in ("HL", "LL")]

    if not highs and not lows:
        return 0.0

    # Count conforming pairs for each possible trend direction
    bull_count = sum(1 for p in highs if p.classification == "HH") + \
                 sum(1 for p in lows if p.classification == "HL")
    bear_count = sum(1 for p in highs if p.classification == "LH") + \
                 sum(1 for p in lows if p.classification == "LL")
    total = len(highs) + len(lows)

    pattern_score = max(bull_count, bear_count) / total if total > 0 else 0.0

    # Small bonus for BOS/CHoCH events (evidence of active structure breaks)
    event_bonus = sum(0.05 if e.kind == "BOS" else 0.10 for e in events)
    event_bonus = min(event_bonus, 0.30)  # cap bonus at 0.30

    return min(pattern_score + event_bonus, 1.0)


def _build_explanation(
    trend_bias: str,
    events: list[StructureEvent],
    swing_points: list[SwingPoint],
) -> str:
    """Build a human-readable summary string."""
    n_bos = sum(1 for e in events if e.kind == "BOS")
    n_choch = sum(1 for e in events if e.kind == "CHoCH")
    n_highs = sum(1 for p in swing_points if p.kind == "high")
    n_lows = sum(1 for p in swing_points if p.kind == "low")

    parts = [
        f"Trend bias: {trend_bias}.",
        f"Swing points: {n_highs} highs, {n_lows} lows.",
        f"Events: {n_bos} BOS, {n_choch} CHoCH.",
    ]
    if events:
        last = events[-1]
        parts.append(f"Last event: {last.kind} {last.direction} at bar {last.index}.")
    return " ".join(parts)

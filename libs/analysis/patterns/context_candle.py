"""
Context-aware candle pattern detectors (12 patterns).

These patterns require 1–5 prior bars for context (average range,
prior high/low reference levels). They complement the classic
single-candle and multi-candle patterns.

Patterns:
    InsideBarDetector, OutsideBarDetector, PinBarDetector,
    RejectionCandleDetector, BreakoutCandleDetector,
    ExhaustionCandleDetector, MomentumCandleDetector,
    LongWickCandleDetector, NarrowRangeCandleDetector,
    WideRangeCandleDetector, TrapCandleDetector,
    FailedBreakoutCandleDetector
"""
from __future__ import annotations

import pandas as pd

from libs.analysis.patterns.base import BasePatternDetector, MatchingMode
from libs.core.models.domain import PatternBias, PatternResult


def _f(row: pd.Series, col: str, default: float = 0.0) -> float:
    return float(row.get(col, default))


def _avg_range(df: pd.DataFrame, n: int = 10) -> float:
    """Average high-low range over last n bars."""
    if len(df) < 2:
        return (float(df.iloc[-1]["high"]) - float(df.iloc[-1]["low"])) if len(df) else 1e-9
    tail = df.tail(n)
    ranges = tail["high"].astype(float) - tail["low"].astype(float)
    return float(ranges.mean()) or 1e-9


# ── Pattern 1: Inside Bar ─────────────────────────────────────────────────────

class InsideBarDetector(BasePatternDetector):
    """
    Inside Bar: current bar fully contained within prior bar's range.
    Signals compression before a breakout. Bias is context-dependent.
    """

    min_bars_required = 2

    @property
    def name(self) -> str:
        return "inside_bar"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2:
            return self._no_pattern()
        try:
            prior = df.iloc[-2]
            curr = df.iloc[-1]
            prior_high = float(prior["high"])
            prior_low = float(prior["low"])
            curr_high = float(curr["high"])
            curr_low = float(curr["low"])
            if not (curr_high <= prior_high and curr_low >= prior_low):
                return self._no_pattern()
            prior_range = prior_high - prior_low
            curr_range = curr_high - curr_low
            if prior_range <= 0:
                return self._no_pattern({"reason": "zero prior range"})
            containment = 1.0 - (curr_range / prior_range)
            conf = min(1.0, 0.50 + containment * 0.25)
            return self._result(
                conf,
                {"containment": round(containment, 4), "curr_range": round(curr_range, 6)},
                PatternBias.NEUTRAL,
                explanation=f"Inside bar: range {curr_range:.4f} contained within prior range {prior_range:.4f} — compression before breakout",
                category="indecision",
                reliability=0.52,
            )
        except Exception:
            return self._no_pattern()


# ── Pattern 2: Outside Bar ────────────────────────────────────────────────────

class OutsideBarDetector(BasePatternDetector):
    """
    Outside Bar: current bar engulfs the entire prior bar's range.
    Signals market expansion; bias determined by close position.
    """

    min_bars_required = 2

    @property
    def name(self) -> str:
        return "outside_bar"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2:
            return self._no_pattern()
        try:
            prior = df.iloc[-2]
            curr = df.iloc[-1]
            prior_high = float(prior["high"])
            prior_low = float(prior["low"])
            curr_high = float(curr["high"])
            curr_low = float(curr["low"])
            curr_close = float(curr["close"])
            if not (curr_high > prior_high and curr_low < prior_low):
                return self._no_pattern()
            prior_range = prior_high - prior_low
            curr_range = curr_high - curr_low
            expansion = curr_range / (prior_range + 1e-9)
            # Bias based on where close sits in current range
            close_position = (curr_close - curr_low) / (curr_range + 1e-9)
            if close_position >= 0.60:
                bias = PatternBias.BULLISH
                bias_str = "bullish"
            elif close_position <= 0.40:
                bias = PatternBias.BEARISH
                bias_str = "bearish"
            else:
                bias = PatternBias.NEUTRAL
                bias_str = "neutral"
            rv = _f(curr, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            conf = min(1.0, 0.45 + vol_bonus + (expansion - 1.0) * 0.05)
            conf = min(0.75, conf)
            return self._result(
                conf,
                {"expansion": round(expansion, 2), "close_position": round(close_position, 3)},
                bias,
                explanation=f"Outside bar: range {expansion:.1f}x prior — market expansion, bias {bias_str}",
                category="continuation",
                reliability=0.48,
            )
        except Exception:
            return self._no_pattern()


# ── Pattern 3: Pin Bar ────────────────────────────────────────────────────────

class PinBarDetector(BasePatternDetector):
    """
    Pin Bar: small body with a long dominant wick — strong rejection of prices.
    Bullish pin has long lower wick; bearish pin has long upper wick.
    """

    min_bars_required = 1

    @property
    def name(self) -> str:
        return "pin_bar"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1:
            return self._no_pattern()
        try:
            curr = df.iloc[-1]
            o = float(curr["open"])
            h = float(curr["high"])
            l = float(curr["low"])
            c = float(curr["close"])
            body = abs(c - o)
            rng = h - l
            if rng <= 0:
                return self._no_pattern({"reason": "zero range"})
            body_pct = body / rng
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            rv = _f(curr, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            # Bullish pin: small body, long lower wick, tiny upper wick
            if body_pct < 0.35 and lower_wick >= 0.60 * rng and upper_wick <= 0.15 * rng:
                conf = min(1.0, 0.55 + min(0.30, (lower_wick / rng - 0.60) * 0.8) + vol_bonus)
                return self._result(
                    conf,
                    {"body_pct": round(body_pct, 4), "lower_wick_pct": round(lower_wick / rng, 4)},
                    PatternBias.BULLISH,
                    explanation=f"Bullish pin bar: lower wick {lower_wick/rng:.0%} of range — strong rejection of lower prices",
                    category="reversal",
                    reliability=0.62,
                )
            # Bearish pin: small body, long upper wick, tiny lower wick
            if body_pct < 0.35 and upper_wick >= 0.60 * rng and lower_wick <= 0.15 * rng:
                conf = min(1.0, 0.55 + min(0.30, (upper_wick / rng - 0.60) * 0.8) + vol_bonus)
                return self._result(
                    conf,
                    {"body_pct": round(body_pct, 4), "upper_wick_pct": round(upper_wick / rng, 4)},
                    PatternBias.BEARISH,
                    explanation=f"Bearish pin bar: upper wick {upper_wick/rng:.0%} of range — strong rejection of higher prices",
                    category="reversal",
                    reliability=0.62,
                )
            return self._no_pattern()
        except Exception:
            return self._no_pattern()


# ── Pattern 4: Rejection Candle ───────────────────────────────────────────────

class RejectionCandleDetector(BasePatternDetector):
    """
    Rejection Candle: wider-body variant of pin bar with confirmed close direction.
    Requires bullish close for bullish rejection, bearish close for bearish rejection.
    """

    min_bars_required = 1

    @property
    def name(self) -> str:
        return "rejection_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1:
            return self._no_pattern()
        try:
            curr = df.iloc[-1]
            o = float(curr["open"])
            h = float(curr["high"])
            l = float(curr["low"])
            c = float(curr["close"])
            body = abs(c - o)
            rng = h - l
            if rng <= 0:
                return self._no_pattern({"reason": "zero range"})
            body_pct = body / rng
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            rv = _f(curr, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            # Bullish rejection: small-ish body, long lower wick, bullish close
            if body_pct < 0.45 and lower_wick >= 0.50 * rng and c > o:
                conf = min(0.85, 0.50 + (lower_wick / rng - 0.50) * 0.4 + vol_bonus)
                return self._result(
                    conf,
                    {"body_pct": round(body_pct, 4), "lower_wick_pct": round(lower_wick / rng, 4)},
                    PatternBias.BULLISH,
                    explanation=f"Bullish rejection: body {body_pct:.0%}, lower wick {lower_wick/rng:.0%} of range with bullish close",
                    category="reversal",
                    reliability=0.58,
                )
            # Bearish rejection: small-ish body, long upper wick, bearish close
            if body_pct < 0.45 and upper_wick >= 0.50 * rng and c < o:
                conf = min(0.85, 0.50 + (upper_wick / rng - 0.50) * 0.4 + vol_bonus)
                return self._result(
                    conf,
                    {"body_pct": round(body_pct, 4), "upper_wick_pct": round(upper_wick / rng, 4)},
                    PatternBias.BEARISH,
                    explanation=f"Bearish rejection: body {body_pct:.0%}, upper wick {upper_wick/rng:.0%} of range with bearish close",
                    category="reversal",
                    reliability=0.58,
                )
            return self._no_pattern()
        except Exception:
            return self._no_pattern()


# ── Pattern 5: Breakout Candle ────────────────────────────────────────────────

class BreakoutCandleDetector(BasePatternDetector):
    """
    Breakout Candle: closes beyond the prior N-bar high/low with an above-average range.
    Signals a genuine directional break with momentum.
    """

    min_bars_required = 5

    @property
    def name(self) -> str:
        return "breakout_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 5:
            return self._no_pattern()
        try:
            avg_r = _avg_range(df, n=10)
            current = df.iloc[-1]
            prior_bars = df.iloc[-6:-1]
            prior_high = float(prior_bars["high"].astype(float).max())
            prior_low = float(prior_bars["low"].astype(float).min())
            curr_high = float(current["high"])
            curr_low = float(current["low"])
            curr_close = float(current["close"])
            curr_open = float(current["open"])
            curr_range = curr_high - curr_low
            curr_body = abs(curr_close - curr_open)
            body_pct = curr_body / (curr_range + 1e-9)
            rv = _f(current, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            # Bullish breakout
            if curr_close > prior_high and body_pct >= 0.55 and curr_range >= avg_r * 1.2:
                conf = min(1.0, 0.55 + min(0.25, (curr_range / avg_r - 1.2) * 0.1) + vol_bonus)
                return self._result(
                    conf,
                    {"prior_high": round(prior_high, 6), "range_ratio": round(curr_range / avg_r, 2)},
                    PatternBias.BULLISH,
                    explanation=f"Breakout candle: closes {curr_close - prior_high:.4f} above prior {len(prior_bars)}-bar high, range {curr_range/avg_r:.1f}x avg",
                    category="continuation",
                    reliability=0.60,
                )
            # Bearish breakout
            if curr_close < prior_low and body_pct >= 0.55 and curr_range >= avg_r * 1.2:
                conf = min(1.0, 0.55 + min(0.25, (curr_range / avg_r - 1.2) * 0.1) + vol_bonus)
                return self._result(
                    conf,
                    {"prior_low": round(prior_low, 6), "range_ratio": round(curr_range / avg_r, 2)},
                    PatternBias.BEARISH,
                    explanation=f"Breakout candle: closes {prior_low - curr_close:.4f} below prior {len(prior_bars)}-bar low, range {curr_range/avg_r:.1f}x avg",
                    category="continuation",
                    reliability=0.60,
                )
            return self._no_pattern()
        except Exception:
            return self._no_pattern()


# ── Pattern 6: Exhaustion Candle ──────────────────────────────────────────────

class ExhaustionCandleDetector(BasePatternDetector):
    """
    Exhaustion Candle: very large bar that closes far from its extreme,
    suggesting the move is losing steam. Counter-trend signal.
    """

    min_bars_required = 5

    @property
    def name(self) -> str:
        return "exhaustion_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 5:
            return self._no_pattern()
        try:
            avg_r = _avg_range(df, n=10)
            current = df.iloc[-1]
            h = float(current["high"])
            l = float(current["low"])
            o = float(current["open"])
            c = float(current["close"])
            curr_range = h - l
            rv = _f(current, "relative_volume", 1.0)
            # Must be a large bar
            if curr_range < avg_r * 1.8:
                return self._no_pattern()
            vol_spike_bonus = 0.10 if rv >= 1.5 else 0.0
            conf_base = min(1.0, 0.50 + min(0.20, (curr_range / avg_r - 1.8) * 0.08) + vol_spike_bonus)
            # Bullish exhaustion (BEARISH signal): bullish bar with long upper wick
            if c > o and (h - c) / curr_range >= 0.25:
                return self._result(
                    conf_base,
                    {"range_ratio": round(curr_range / avg_r, 2), "upper_wick_pct": round((h - c) / curr_range, 4)},
                    PatternBias.BEARISH,
                    explanation=f"Bullish exhaustion: range {curr_range/avg_r:.1f}x avg with {(h-c)/curr_range:.0%} upper wick — buyers potentially exhausted",
                    category="reversal",
                    reliability=0.55,
                )
            # Bearish exhaustion (BULLISH signal): bearish bar with long lower wick
            if c < o and (c - l) / curr_range >= 0.25:
                return self._result(
                    conf_base,
                    {"range_ratio": round(curr_range / avg_r, 2), "lower_wick_pct": round((c - l) / curr_range, 4)},
                    PatternBias.BULLISH,
                    explanation=f"Bearish exhaustion: range {curr_range/avg_r:.1f}x avg with {(c-l)/curr_range:.0%} lower wick — sellers potentially exhausted",
                    category="reversal",
                    reliability=0.55,
                )
            return self._no_pattern()
        except Exception:
            return self._no_pattern()


# ── Pattern 7: Momentum Candle ────────────────────────────────────────────────

class MomentumCandleDetector(BasePatternDetector):
    """
    Momentum Candle: large-bodied bar in the direction of the move.
    Signals strong directional intent with continuation bias.
    """

    min_bars_required = 3

    @property
    def name(self) -> str:
        return "momentum_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3:
            return self._no_pattern()
        try:
            avg_r = _avg_range(df, n=10)
            current = df.iloc[-1]
            h = float(current["high"])
            l = float(current["low"])
            o = float(current["open"])
            c = float(current["close"])
            curr_range = h - l
            curr_body = abs(c - o)
            body_pct = curr_body / (curr_range + 1e-9)
            rv = _f(current, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            if not (body_pct >= 0.65 and curr_range >= avg_r * 0.8):
                return self._no_pattern()
            conf = min(1.0, 0.50 + min(0.25, body_pct * 0.3) + vol_bonus)
            if c > o:
                return self._result(
                    conf,
                    {"body_pct": round(body_pct, 4), "range_ratio": round(curr_range / avg_r, 2)},
                    PatternBias.BULLISH,
                    explanation=f"Bullish momentum: {body_pct:.0%} body, range {curr_range/avg_r:.1f}x avg — strong directional move",
                    category="continuation",
                    reliability=0.58,
                )
            if c < o:
                return self._result(
                    conf,
                    {"body_pct": round(body_pct, 4), "range_ratio": round(curr_range / avg_r, 2)},
                    PatternBias.BEARISH,
                    explanation=f"Bearish momentum: {body_pct:.0%} body, range {curr_range/avg_r:.1f}x avg — strong directional move",
                    category="continuation",
                    reliability=0.58,
                )
            return self._no_pattern()
        except Exception:
            return self._no_pattern()


# ── Pattern 8: Long Wick Candle ───────────────────────────────────────────────

class LongWickCandleDetector(BasePatternDetector):
    """
    Long Wick Candle: total wicks dwarf the body — significant price rejection
    on both sides. Signals indecision with potential reversal.
    """

    min_bars_required = 1

    @property
    def name(self) -> str:
        return "long_wick_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1:
            return self._no_pattern()
        try:
            curr = df.iloc[-1]
            h = float(curr["high"])
            l = float(curr["low"])
            o = float(curr["open"])
            c = float(curr["close"])
            rng = h - l
            if rng <= 0:
                return self._no_pattern({"reason": "zero range"})
            body = abs(c - o)
            if body == 0:
                body = 0.001 * rng
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            total_wick = upper_wick + lower_wick
            wick_to_body = total_wick / (body + 1e-9)
            if wick_to_body < 2.5:
                return self._no_pattern({"wick_to_body": round(wick_to_body, 2)})
            conf = min(1.0, 0.42 + min(0.20, (wick_to_body - 2.5) * 0.05))
            return self._result(
                conf,
                {"wick_to_body": round(wick_to_body, 2)},
                PatternBias.NEUTRAL,
                explanation=f"Long wick candle: total wicks {wick_to_body:.1f}x body — significant price rejection on both sides",
                category="indecision",
                reliability=0.42,
            )
        except Exception:
            return self._no_pattern()


# ── Pattern 9: Narrow Range Candle ───────────────────────────────────────────

class NarrowRangeCandleDetector(BasePatternDetector):
    """
    Narrow Range Candle: range well below recent average — compression that
    typically precedes a volatility expansion.
    """

    min_bars_required = 5

    @property
    def name(self) -> str:
        return "narrow_range_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 5:
            return self._no_pattern()
        try:
            avg_r = _avg_range(df, n=min(10, len(df) - 1))
            curr = df.iloc[-1]
            curr_range = float(curr["high"]) - float(curr["low"])
            if curr_range >= avg_r * 0.60:
                return self._no_pattern({"range_ratio": round(curr_range / avg_r, 3)})
            conf = min(1.0, 0.45 + min(0.20, (1.0 - curr_range / avg_r) * 0.30))
            return self._result(
                conf,
                {"range_ratio": round(curr_range / avg_r, 4)},
                PatternBias.NEUTRAL,
                explanation=f"Narrow range: {curr_range/avg_r:.0%} of avg range — compression, potential volatility expansion ahead",
                category="indecision",
                reliability=0.45,
            )
        except Exception:
            return self._no_pattern()


# ── Pattern 10: Wide Range Candle ─────────────────────────────────────────────

class WideRangeCandleDetector(BasePatternDetector):
    """
    Wide Range Candle: range well above recent average — strong directional intent.
    Bias follows the candle direction.
    """

    min_bars_required = 5

    @property
    def name(self) -> str:
        return "wide_range_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 5:
            return self._no_pattern()
        try:
            avg_r = _avg_range(df, n=min(10, len(df) - 1))
            current = df.iloc[-1]
            h = float(current["high"])
            l = float(current["low"])
            o = float(current["open"])
            c = float(current["close"])
            curr_range = h - l
            if curr_range < avg_r * 1.80:
                return self._no_pattern({"range_ratio": round(curr_range / avg_r, 3)})
            if c > o:
                bias = PatternBias.BULLISH
                bias_str = "bullish"
            elif c < o:
                bias = PatternBias.BEARISH
                bias_str = "bearish"
            else:
                bias = PatternBias.NEUTRAL
                bias_str = "neutral"
            rv = _f(current, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            conf = min(1.0, 0.48 + min(0.25, (curr_range / avg_r - 1.80) * 0.08) + vol_bonus)
            return self._result(
                conf,
                {"range_ratio": round(curr_range / avg_r, 2)},
                bias,
                explanation=f"Wide range candle: {curr_range/avg_r:.1f}x avg range — strong directional intent, bias {bias_str}",
                category="continuation",
                reliability=0.52,
            )
        except Exception:
            return self._no_pattern()


# ── Pattern 11: Trap Candle ───────────────────────────────────────────────────

class TrapCandleDetector(BasePatternDetector):
    """
    Trap Candle: bar opens beyond a prior level (false breakout/breakdown)
    but closes back on the other side — trapping late movers.
    """

    min_bars_required = 3

    @property
    def name(self) -> str:
        return "trap_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3:
            return self._no_pattern()
        try:
            current = df.iloc[-1]
            curr_open = float(current["open"])
            curr_close = float(current["close"])
            # Collect prior bars (excluding current)
            lookback_count = min(5, len(df) - 1)
            prior_slice = df.iloc[-lookback_count - 1:-1]
            lookback_high = float(prior_slice["high"].astype(float).max())
            lookback_low = float(prior_slice["low"].astype(float).min())
            rv = _f(current, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            conf = min(1.0, 0.55 + vol_bonus)
            # Bull trap (BEARISH): opened above high, closed back below it
            if curr_open > lookback_high and curr_close < lookback_high:
                return self._result(
                    conf,
                    {"lookback_high": round(lookback_high, 6), "open": round(curr_open, 6)},
                    PatternBias.BEARISH,
                    explanation=f"Bull trap: opened {curr_open - lookback_high:.4f} above prior high then reversed — buyers trapped above resistance",
                    category="reversal",
                    reliability=0.60,
                )
            # Bear trap (BULLISH): opened below low, closed back above it
            if curr_open < lookback_low and curr_close > lookback_low:
                return self._result(
                    conf,
                    {"lookback_low": round(lookback_low, 6), "open": round(curr_open, 6)},
                    PatternBias.BULLISH,
                    explanation=f"Bear trap: opened {lookback_low - curr_open:.4f} below prior low then reversed — sellers trapped below support",
                    category="reversal",
                    reliability=0.60,
                )
            return self._no_pattern()
        except Exception:
            return self._no_pattern()


# ── Pattern 12: Failed Breakout Candle ───────────────────────────────────────

class FailedBreakoutCandleDetector(BasePatternDetector):
    """
    Failed Breakout Candle: a prior bar broke above/below a level, but the
    current bar closes back on the other side — confirming the failure.
    """

    min_bars_required = 4

    @property
    def name(self) -> str:
        return "failed_breakout_candle"

    @property
    def bias(self) -> PatternBias:
        return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 4:
            return self._no_pattern()
        try:
            bar_minus_1 = df.iloc[-1]   # current
            bar_minus_2 = df.iloc[-2]   # prior
            lookback_bars = df.iloc[-min(5, len(df)):-2]
            if len(lookback_bars) == 0:
                return self._no_pattern()
            lb_high = float(lookback_bars["high"].astype(float).max())
            lb_low = float(lookback_bars["low"].astype(float).min())
            curr_close = float(bar_minus_1["close"])
            prior_close = float(bar_minus_2["close"])
            rv = _f(bar_minus_1, "relative_volume", 1.0)
            vol_bonus = self._vol_bonus(rv)
            conf = min(1.0, 0.57 + vol_bonus)
            # Failed bull breakout (BEARISH): prior closed above lb_high, current reversed below
            if prior_close > lb_high and curr_close < lb_high:
                return self._result(
                    conf,
                    {"lb_high": round(lb_high, 6), "prior_close": round(prior_close, 6)},
                    PatternBias.BEARISH,
                    explanation=f"Failed breakout: prior bar closed above {lb_high:.4f} but current reversed below — false breakout, potential reversal",
                    category="reversal",
                    reliability=0.62,
                )
            # Failed bear breakout (BULLISH): prior closed below lb_low, current reversed above
            if prior_close < lb_low and curr_close > lb_low:
                return self._result(
                    conf,
                    {"lb_low": round(lb_low, 6), "prior_close": round(prior_close, 6)},
                    PatternBias.BULLISH,
                    explanation=f"Failed breakdown: prior bar closed below {lb_low:.4f} but current reversed above — false breakdown, potential reversal",
                    category="reversal",
                    reliability=0.62,
                )
            return self._no_pattern()
        except Exception:
            return self._no_pattern()

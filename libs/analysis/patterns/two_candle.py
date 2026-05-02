"""
Two-candle pattern detectors (8 patterns).

Bullish/Bearish Engulfing, Bullish/Bearish Harami,
Piercing Line, Dark Cloud Cover, Tweezer Top/Bottom.
"""
from __future__ import annotations

import pandas as pd

from libs.analysis.patterns.base import BasePatternDetector, MatchingMode
from libs.core.models.domain import PatternBias, PatternResult


def _two(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    return df.iloc[-2], df.iloc[-1]


def _f(row: pd.Series, col: str, default: float = 0.0) -> float:
    return float(row.get(col, default))


# ── Engulfing ─────────────────────────────────────────────────────────────────

class BullishEngulfingDetector(BasePatternDetector):
    min_bars_required = 2

    @property
    def name(self) -> str: return "bullish_engulfing"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        if not (float(p["close"]) < float(p["open"]) and float(c["close"]) > float(c["open"])):
            return self._no_pattern({"reason": "direction mismatch"})
        if not (float(c["open"]) <= float(p["close"]) and float(c["close"]) >= float(p["open"])):
            return self._no_pattern({"reason": "no engulfment"})
        prev_body = abs(float(p["open"]) - float(p["close"]))
        curr_body = abs(float(c["open"]) - float(c["close"]))
        ratio = curr_body / (prev_body + 1e-9)
        rv = _f(c, "relative_volume", 1.0)
        conf = min(1.0, 0.55 + min(0.30, (ratio - 1.0) * 0.15) + self._vol_bonus(rv))
        return self._result(
            conf,
            {"engulf_ratio": round(ratio, 2)},
            category="reversal",
            reliability=0.63,
            explanation=f"Bullish engulfing: current body {ratio:.1f}x prior — buyers overwhelm prior sellers",
        )


class BearishEngulfingDetector(BasePatternDetector):
    min_bars_required = 2

    @property
    def name(self) -> str: return "bearish_engulfing"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        if not (float(p["close"]) > float(p["open"]) and float(c["close"]) < float(c["open"])):
            return self._no_pattern()
        if not (float(c["open"]) >= float(p["close"]) and float(c["close"]) <= float(p["open"])):
            return self._no_pattern()
        prev_body = abs(float(p["open"]) - float(p["close"]))
        curr_body = abs(float(c["open"]) - float(c["close"]))
        ratio = curr_body / (prev_body + 1e-9)
        rv = _f(c, "relative_volume", 1.0)
        conf = min(1.0, 0.55 + min(0.30, (ratio - 1.0) * 0.15) + self._vol_bonus(rv))
        return self._result(
            conf,
            {"engulf_ratio": round(ratio, 2)},
            category="reversal",
            reliability=0.63,
            explanation=f"Bearish engulfing: current body {ratio:.1f}x prior — sellers overwhelm prior buyers",
        )


# ── Harami ────────────────────────────────────────────────────────────────────

class BullishHaramiDetector(BasePatternDetector):
    min_bars_required = 2

    @property
    def name(self) -> str: return "bullish_harami"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        if not (float(p["close"]) < float(p["open"]) and float(c["close"]) > float(c["open"])):
            return self._no_pattern()
        p_top, p_bot = float(p["open"]), float(p["close"])
        c_top, c_bot = float(c["close"]), float(c["open"])
        if not (c_bot >= p_bot and c_top <= p_top):
            return self._no_pattern({"reason": "not contained"})
        p_body = p_top - p_bot
        c_body = c_top - c_bot
        contain = c_body / (p_body + 1e-9)
        conf = min(1.0, 0.50 + (1.0 - contain) * 0.35)
        return self._result(
            conf,
            {"containment": round(contain, 3)},
            category="reversal",
            reliability=0.50,
            explanation="Bullish harami: small bullish bar inside prior bearish — momentum potentially exhausted",
        )


class BearishHaramiDetector(BasePatternDetector):
    min_bars_required = 2

    @property
    def name(self) -> str: return "bearish_harami"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        if not (float(p["close"]) > float(p["open"]) and float(c["close"]) < float(c["open"])):
            return self._no_pattern()
        p_bot, p_top = float(p["open"]), float(p["close"])
        c_top, c_bot = float(c["open"]), float(c["close"])
        if not (c_bot >= p_bot and c_top <= p_top):
            return self._no_pattern()
        p_body = p_top - p_bot
        c_body = c_top - c_bot
        contain = c_body / (p_body + 1e-9)
        conf = min(1.0, 0.50 + (1.0 - contain) * 0.35)
        return self._result(
            conf,
            {"containment": round(contain, 3)},
            category="reversal",
            reliability=0.50,
            explanation="Bearish harami: small bearish bar inside prior bullish — momentum potentially exhausted",
        )


# ── Piercing Line ─────────────────────────────────────────────────────────────

class PiercingLineDetector(BasePatternDetector):
    """
    Piercing Line: bearish candle followed by bullish that opens below low
    and closes above the midpoint of the bearish body.
    """
    min_bars_required = 2

    @property
    def name(self) -> str: return "piercing_line"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        if not (float(p["close"]) < float(p["open"]) and float(c["close"]) > float(c["open"])):
            return self._no_pattern()
        # Current opens below prior low
        if float(c["open"]) >= float(p["low"]):
            return self._no_pattern()
        # Current close penetrates 50%+ into prior bearish body
        mid = (float(p["open"]) + float(p["close"])) / 2
        penetration = self._threshold(0.50)
        if float(c["close"]) < mid:
            return self._no_pattern({"reason": "insufficient penetration"})
        pen_score = (float(c["close"]) - mid) / (float(p["open"]) - mid + 1e-9)
        conf = min(1.0, 0.55 + pen_score * 0.25 + self._vol_bonus(_f(c, "relative_volume", 1.0)))
        return self._result(
            conf,
            {"penetration": round(pen_score, 3)},
            category="reversal",
            reliability=0.58,
            explanation=f"Piercing line: bullish bar closes {pen_score:.0%} into prior bearish body — buyers reclaiming ground",
        )


# ── Dark Cloud Cover ──────────────────────────────────────────────────────────

class DarkCloudCoverDetector(BasePatternDetector):
    """
    Dark Cloud Cover: bullish candle followed by bearish that opens above high
    and closes below the midpoint of the bullish body.
    """
    min_bars_required = 2

    @property
    def name(self) -> str: return "dark_cloud_cover"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        if not (float(p["close"]) > float(p["open"]) and float(c["close"]) < float(c["open"])):
            return self._no_pattern()
        if float(c["open"]) <= float(p["high"]):
            return self._no_pattern()
        mid = (float(p["open"]) + float(p["close"])) / 2
        if float(c["close"]) > mid:
            return self._no_pattern()
        pen_score = (mid - float(c["close"])) / (mid - float(p["open"]) + 1e-9)
        conf = min(1.0, 0.55 + pen_score * 0.25 + self._vol_bonus(_f(c, "relative_volume", 1.0)))
        return self._result(
            conf,
            {"penetration": round(pen_score, 3)},
            category="reversal",
            reliability=0.58,
            explanation=f"Dark cloud cover: bearish bar closes {pen_score:.0%} into prior bullish body — sellers reclaiming ground",
        )


# ── Tweezers ──────────────────────────────────────────────────────────────────

class TweezerBottomDetector(BasePatternDetector):
    """Tweezer Bottom: matching lows at support. First bearish, second bullish."""
    min_bars_required = 2
    _TOLERANCE = 0.002

    @property
    def name(self) -> str: return "tweezer_bottom"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        low1, low2 = float(p["low"]), float(c["low"])
        avg = (low1 + low2) / 2
        diff_pct = abs(low1 - low2) / (avg + 1e-9)
        tol = self._threshold(self._TOLERANCE)
        if diff_pct > tol: return self._no_pattern({"low_diff": round(diff_pct, 4)})
        if not (float(p["close"]) < float(p["open"]) and float(c["close"]) > float(c["open"])):
            return self._no_pattern()
        precision = 1.0 - diff_pct / (tol + 1e-9)
        conf = min(1.0, 0.55 + precision * 0.30)
        return self._result(
            conf,
            {"level": round(avg, 4)},
            category="reversal",
            reliability=0.55,
            explanation=f"Tweezer bottom: matching lows at {avg:.4f} — failed breakdown, support confirmed",
        )


class TweezerTopDetector(BasePatternDetector):
    """Tweezer Top: matching highs at resistance. First bullish, second bearish."""
    min_bars_required = 2
    _TOLERANCE = 0.002

    @property
    def name(self) -> str: return "tweezer_top"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 2: return self._no_pattern()
        p, c = _two(df)
        h1, h2 = float(p["high"]), float(c["high"])
        avg = (h1 + h2) / 2
        diff_pct = abs(h1 - h2) / (avg + 1e-9)
        tol = self._threshold(self._TOLERANCE)
        if diff_pct > tol: return self._no_pattern()
        if not (float(p["close"]) > float(p["open"]) and float(c["close"]) < float(c["open"])):
            return self._no_pattern()
        precision = 1.0 - diff_pct / (tol + 1e-9)
        conf = min(1.0, 0.55 + precision * 0.30)
        return self._result(
            conf,
            {"level": round(avg, 4)},
            category="reversal",
            reliability=0.55,
            explanation=f"Tweezer top: matching highs at {avg:.4f} — failed breakout, resistance confirmed",
        )

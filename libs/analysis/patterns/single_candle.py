"""
Single-candle pattern detectors (10 patterns).

Hammer, Inverted Hammer, Shooting Star, Hanging Man,
Doji (standard / Dragonfly / Gravestone),
Spinning Top, Bullish Marubozu, Bearish Marubozu.
"""
from __future__ import annotations

import pandas as pd

from libs.analysis.patterns.base import BasePatternDetector, MatchingMode
from libs.core.models.domain import PatternBias, PatternResult


def _row(df: pd.DataFrame, idx: int = -1) -> pd.Series:
    return df.iloc[idx]


def _vals(row: pd.Series) -> tuple[float, float, float, float, float, float, float, float, float]:
    o = float(row["open"])
    h = float(row["high"])
    l = float(row["low"])
    c = float(row["close"])
    body = float(row.get("body_size", abs(c - o)))
    rng = float(row.get("total_range", h - l)) or 1e-9
    uw = float(row.get("upper_wick", h - max(o, c)))
    lw = float(row.get("lower_wick", min(o, c) - l))
    rv = float(row.get("relative_volume", 1.0))
    return o, h, l, c, body, rng, uw, lw, rv


# ── Hammer ────────────────────────────────────────────────────────────────────

class HammerDetector(BasePatternDetector):
    """Hammer: small body at top, long lower wick (>=2x body), tiny upper wick."""

    min_bars_required = 1

    @property
    def name(self) -> str: return "hammer"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        if body == 0: return self._no_pattern({"reason": "zero body"})
        lw_ratio = lw / body
        uw_ratio = uw / body
        min_lw = self._threshold(2.0)
        max_uw = 1.0 / self._threshold(1.0)
        if lw_ratio < min_lw or uw_ratio > max_uw * 0.5:
            return self._no_pattern({"lw_ratio": round(lw_ratio, 2)})
        conf = min(1.0, 0.55 + min(0.30, (lw_ratio - min_lw) / 3) + self._vol_bonus(rv))
        return self._result(
            conf,
            {"lw_ratio": round(lw_ratio, 2), "rv": round(rv, 2)},
            category="reversal",
            reliability=0.60,
            explanation=f"Hammer: lower wick {lw_ratio:.1f}x body — rejection of lower prices",
        )


class InvertedHammerDetector(BasePatternDetector):
    """Inverted Hammer: small body at bottom, long upper wick. Bullish at support."""

    min_bars_required = 1

    @property
    def name(self) -> str: return "inverted_hammer"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        if body == 0: return self._no_pattern()
        uw_ratio = uw / body
        lw_ratio = lw / body
        if uw_ratio < self._threshold(2.0) or lw_ratio > 0.5:
            return self._no_pattern()
        conf = min(1.0, 0.50 + min(0.30, (uw_ratio - 2.0) / 3) + self._vol_bonus(rv))
        return self._result(
            conf,
            category="reversal",
            reliability=0.55,
            explanation=f"Inverted hammer: upper wick {uw_ratio:.1f}x body — potential buying pressure from below",
        )


# ── Shooting Star ─────────────────────────────────────────────────────────────

class ShootingStarDetector(BasePatternDetector):
    """Shooting Star: small body at bottom of range, long upper wick. Bearish at resistance."""

    min_bars_required = 1

    @property
    def name(self) -> str: return "shooting_star"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        if body == 0: return self._no_pattern()
        uw_ratio = uw / body
        lw_ratio = lw / body
        if uw_ratio < self._threshold(2.0) or lw_ratio > 0.5:
            return self._no_pattern()
        conf = min(1.0, 0.55 + min(0.30, (uw_ratio - 2.0) / 3) + self._vol_bonus(rv))
        return self._result(
            conf,
            category="reversal",
            reliability=0.60,
            explanation=f"Shooting star: upper wick {uw_ratio:.1f}x body — rejection of higher prices",
        )


# ── Hanging Man ───────────────────────────────────────────────────────────────

class HangingManDetector(BasePatternDetector):
    """Hanging Man: same shape as Hammer but at the TOP of an uptrend. Bearish."""

    min_bars_required = 1

    @property
    def name(self) -> str: return "hanging_man"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        if body == 0: return self._no_pattern()
        lw_ratio = lw / body
        uw_ratio = uw / body
        if lw_ratio < self._threshold(2.0) or uw_ratio > 0.5:
            return self._no_pattern()
        # Hanging man is slightly less reliable than hammer — lower base confidence
        conf = min(1.0, 0.50 + min(0.25, (lw_ratio - 2.0) / 3) + self._vol_bonus(rv))
        return self._result(
            conf,
            category="reversal",
            reliability=0.50,
            explanation=f"Hanging man: lower wick {lw_ratio:.1f}x body — potential selling pressure from above",
        )


# ── Doji family ───────────────────────────────────────────────────────────────

class DojiDetector(BasePatternDetector):
    """
    Doji: body < 10% of range (standard doji).
    Also detects Dragonfly (long lower wick) and Gravestone (long upper wick).
    """

    min_bars_required = 1

    @property
    def name(self) -> str: return "doji"

    @property
    def bias(self) -> PatternBias: return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        body_pct = float(_row(df).get("body_pct", body / rng))
        if body_pct >= 0.10: return self._no_pattern({"body_pct": round(body_pct, 3)})

        # Subtype
        wick_ratio = uw / (lw + 1e-9)
        if wick_ratio > 3.0:
            subtype, bias = "gravestone", PatternBias.BEARISH
        elif wick_ratio < 0.33:
            subtype, bias = "dragonfly", PatternBias.BULLISH
        else:
            subtype, bias = "standard", PatternBias.NEUTRAL

        conf = min(1.0, 0.50 + (0.10 - body_pct) * 5)
        return self._result(
            conf,
            {"subtype": subtype, "body_pct": round(body_pct, 4)},
            bias,
            category="indecision",
            reliability=0.45,
            explanation=f"Doji ({subtype}): body {body_pct:.1%} of range — buyers and sellers in equilibrium",
        )


class DragonflyDojiDetector(BasePatternDetector):
    """Explicit Dragonfly Doji detector."""
    min_bars_required = 1

    @property
    def name(self) -> str: return "dragonfly_doji"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        base = DojiDetector(self.mode).detect(df)
        if not base.detected: return self._no_pattern()
        if base.details.get("subtype") != "dragonfly": return self._no_pattern()
        return self._result(
            base.confidence,
            category="reversal",
            reliability=0.55,
            explanation="Dragonfly doji: long lower wick, negligible upper — strong support rejection",
        )


class GravestoneDojiDetector(BasePatternDetector):
    """Explicit Gravestone Doji detector."""
    min_bars_required = 1

    @property
    def name(self) -> str: return "gravestone_doji"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        base = DojiDetector(self.mode).detect(df)
        if not base.detected: return self._no_pattern()
        if base.details.get("subtype") != "gravestone": return self._no_pattern()
        return self._result(
            base.confidence,
            category="reversal",
            reliability=0.55,
            explanation="Gravestone doji: long upper wick, negligible lower — strong resistance rejection",
        )


# ── Spinning Top ──────────────────────────────────────────────────────────────

class SpinningTopDetector(BasePatternDetector):
    """Spinning Top: small body, significant wicks on both sides. Indecision."""

    min_bars_required = 1

    @property
    def name(self) -> str: return "spinning_top"

    @property
    def bias(self) -> PatternBias: return PatternBias.NEUTRAL

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        body_pct = body / rng
        if not (0.10 <= body_pct <= 0.40):
            return self._no_pattern()
        # Both wicks must be significant
        uw_pct = uw / rng
        lw_pct = lw / rng
        if uw_pct < 0.20 or lw_pct < 0.20:
            return self._no_pattern()
        conf = min(1.0, 0.45 + min(uw_pct, lw_pct) * 0.8)
        return self._result(
            conf,
            category="indecision",
            reliability=0.40,
            explanation="Spinning top: balanced wicks on both sides — market indecision, await confirmation",
        )


# ── Marubozu ─────────────────────────────────────────────────────────────────

class BullishMarubozuDetector(BasePatternDetector):
    """Bullish Marubozu: full-body green candle, minimal wicks. Strong momentum."""

    min_bars_required = 1

    @property
    def name(self) -> str: return "bullish_marubozu"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        if c <= o: return self._no_pattern({"reason": "not bullish"})
        body_pct = body / rng
        if body_pct < self._threshold(0.80):
            return self._no_pattern({"body_pct": round(body_pct, 3)})
        conf = min(1.0, 0.60 + (body_pct - 0.80) * 1.5 + self._vol_bonus(rv))
        return self._result(
            conf,
            {"body_pct": round(body_pct, 3)},
            category="continuation",
            reliability=0.65,
            explanation=f"Bullish marubozu: {body_pct:.0%} body, minimal wicks — strong buying momentum",
        )


class BearishMarubozuDetector(BasePatternDetector):
    """Bearish Marubozu: full-body red candle, minimal wicks. Strong selling."""

    min_bars_required = 1

    @property
    def name(self) -> str: return "bearish_marubozu"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 1: return self._no_pattern()
        o, h, l, c, body, rng, uw, lw, rv = _vals(_row(df))
        if c >= o: return self._no_pattern({"reason": "not bearish"})
        body_pct = body / rng
        if body_pct < self._threshold(0.80):
            return self._no_pattern({"body_pct": round(body_pct, 3)})
        conf = min(1.0, 0.60 + (body_pct - 0.80) * 1.5 + self._vol_bonus(rv))
        return self._result(
            conf,
            {"body_pct": round(body_pct, 3)},
            category="continuation",
            reliability=0.65,
            explanation=f"Bearish marubozu: {body_pct:.0%} body, minimal wicks — strong selling momentum",
        )

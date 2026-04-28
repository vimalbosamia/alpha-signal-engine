"""
Multi-candle pattern detectors (10 patterns).

3-bar: Morning Star, Evening Star, Morning Doji Star, Evening Doji Star,
       Three White Soldiers, Three Black Crows,
       Bullish Abandoned Baby, Bearish Abandoned Baby.
5-bar: Rising Three Methods, Falling Three Methods.
"""
from __future__ import annotations

import pandas as pd

from libs.analysis.patterns.base import BasePatternDetector, MatchingMode
from libs.core.models.domain import PatternBias, PatternResult


def _f(row: pd.Series, col: str, default: float = 0.0) -> float:
    return float(row.get(col, default))


def _body(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def _body_pct(row: pd.Series) -> float:
    rng = float(row["high"]) - float(row["low"])
    return _body(row) / rng if rng > 0 else 0.0


def _is_bullish(row: pd.Series) -> bool:
    return float(row["close"]) > float(row["open"])


def _is_bearish(row: pd.Series) -> bool:
    return float(row["close"]) < float(row["open"])


def _is_doji(row: pd.Series) -> bool:
    return _body_pct(row) < 0.10


# ── Morning Star ──────────────────────────────────────────────────────────────

class MorningStarDetector(BasePatternDetector):
    """
    Morning Star: large bearish | small star | large bullish (>50% into B1).
    Most reliable 3-bar reversal pattern.
    """
    min_bars_required = 3

    @property
    def name(self) -> str: return "morning_star"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if not (_is_bearish(b1) and _is_bullish(b3)): return self._no_pattern()
        if _body_pct(b1) < self._threshold(0.50): return self._no_pattern()
        if _body_pct(b3) < self._threshold(0.50): return self._no_pattern()

        b1_mid = (float(b1["open"]) + float(b1["close"])) / 2
        if float(b3["close"]) < b1_mid: return self._no_pattern({"reason": "insufficient penetration"})

        star_score = max(0.0, 1.0 - _body_pct(b2) * 2)
        pen = (float(b3["close"]) - b1_mid) / (float(b1["open"]) - b1_mid + 1e-9)
        pen_score = min(1.0, pen)
        vol_b = self._vol_bonus(_f(b3, "relative_volume", 1.0))
        conf = min(1.0, 0.60 + star_score * 0.15 + pen_score * 0.15 + vol_b)
        return self._result(conf, {"star_body_pct": round(_body_pct(b2), 3)})


# ── Evening Star ──────────────────────────────────────────────────────────────

class EveningStarDetector(BasePatternDetector):
    """Evening Star: large bullish | small star | large bearish (>50% into B1)."""
    min_bars_required = 3

    @property
    def name(self) -> str: return "evening_star"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if not (_is_bullish(b1) and _is_bearish(b3)): return self._no_pattern()
        if _body_pct(b1) < self._threshold(0.50): return self._no_pattern()
        if _body_pct(b3) < self._threshold(0.50): return self._no_pattern()

        b1_mid = (float(b1["open"]) + float(b1["close"])) / 2
        if float(b3["close"]) > b1_mid: return self._no_pattern()

        star_score = max(0.0, 1.0 - _body_pct(b2) * 2)
        pen = (b1_mid - float(b3["close"])) / (b1_mid - float(b1["open"]) + 1e-9)
        pen_score = min(1.0, pen)
        vol_b = self._vol_bonus(_f(b3, "relative_volume", 1.0))
        conf = min(1.0, 0.60 + star_score * 0.15 + pen_score * 0.15 + vol_b)
        return self._result(conf, {"star_body_pct": round(_body_pct(b2), 3)})


# ── Morning Doji Star ─────────────────────────────────────────────────────────

class MorningDojiStarDetector(BasePatternDetector):
    """Morning Doji Star: Morning Star where B2 is specifically a Doji."""
    min_bars_required = 3

    @property
    def name(self) -> str: return "morning_doji_star"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b2 = df.iloc[-2]
        if not _is_doji(b2): return self._no_pattern({"reason": "B2 not a doji"})
        base = MorningStarDetector(self.mode).detect(df)
        if not base.detected: return self._no_pattern()
        # Doji star is more powerful — boost confidence slightly
        return self._result(min(1.0, base.confidence + 0.05))


# ── Evening Doji Star ─────────────────────────────────────────────────────────

class EveningDojiStarDetector(BasePatternDetector):
    """Evening Doji Star: Evening Star where B2 is specifically a Doji."""
    min_bars_required = 3

    @property
    def name(self) -> str: return "evening_doji_star"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b2 = df.iloc[-2]
        if not _is_doji(b2): return self._no_pattern()
        base = EveningStarDetector(self.mode).detect(df)
        if not base.detected: return self._no_pattern()
        return self._result(min(1.0, base.confidence + 0.05))


# ── Three White Soldiers ──────────────────────────────────────────────────────

class ThreeWhiteSoldiersDetector(BasePatternDetector):
    """Three consecutive bullish candles, each opening within prior body, closing near high."""
    min_bars_required = 3

    @property
    def name(self) -> str: return "three_white_soldiers"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if not all(_is_bullish(b) for b in [b1, b2, b3]):
            return self._no_pattern({"reason": "not all bullish"})
        if not (float(b1["open"]) <= float(b2["open"]) <= float(b1["close"])):
            return self._no_pattern()
        if not (float(b2["open"]) <= float(b3["open"]) <= float(b2["close"])):
            return self._no_pattern()

        def _close_near_high(b: pd.Series) -> float:
            rng = float(b["high"]) - float(b["low"])
            return (float(b["close"]) - float(b["low"])) / rng if rng > 0 else 0

        avg_score = sum(_close_near_high(b) for b in [b1, b2, b3]) / 3
        if avg_score < self._threshold(0.60): return self._no_pattern()
        conf = min(1.0, 0.55 + avg_score * 0.35)
        return self._result(conf, {"avg_close_near_high": round(avg_score, 3)})


# ── Three Black Crows ─────────────────────────────────────────────────────────

class ThreeBlackCrowsDetector(BasePatternDetector):
    """Three consecutive bearish candles opening within prior body, closing near low."""
    min_bars_required = 3

    @property
    def name(self) -> str: return "three_black_crows"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if not all(_is_bearish(b) for b in [b1, b2, b3]):
            return self._no_pattern()
        if not (float(b1["close"]) <= float(b2["open"]) <= float(b1["open"])):
            return self._no_pattern()
        if not (float(b2["close"]) <= float(b3["open"]) <= float(b2["open"])):
            return self._no_pattern()

        def _close_near_low(b: pd.Series) -> float:
            rng = float(b["high"]) - float(b["low"])
            return 1.0 - (float(b["close"]) - float(b["low"])) / rng if rng > 0 else 0

        avg_score = sum(_close_near_low(b) for b in [b1, b2, b3]) / 3
        if avg_score < self._threshold(0.60): return self._no_pattern()
        conf = min(1.0, 0.55 + avg_score * 0.35)
        return self._result(conf)


# ── Bullish Abandoned Baby ────────────────────────────────────────────────────

class BullishAbandonedBabyDetector(BasePatternDetector):
    """
    Bullish Abandoned Baby: large bearish | gapped-down doji | large bullish with gap up.
    Rare but very high reliability.
    """
    min_bars_required = 3

    @property
    def name(self) -> str: return "bullish_abandoned_baby"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if not (_is_bearish(b1) and _is_bullish(b3)): return self._no_pattern()
        if not _is_doji(b2): return self._no_pattern({"reason": "B2 not doji"})

        # B2 gaps down from B1
        if float(b2["high"]) >= float(b1["low"]): return self._no_pattern({"reason": "no gap down"})
        # B3 gaps up from B2
        if float(b3["low"]) <= float(b2["high"]): return self._no_pattern({"reason": "no gap up"})

        conf = min(1.0, 0.80 + self._vol_bonus(_f(b3, "relative_volume", 1.0)))
        return self._result(conf)


# ── Bearish Abandoned Baby ────────────────────────────────────────────────────

class BearishAbandonedBabyDetector(BasePatternDetector):
    """Bearish Abandoned Baby: large bullish | gapped-up doji | large bearish with gap down."""
    min_bars_required = 3

    @property
    def name(self) -> str: return "bearish_abandoned_baby"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 3: return self._no_pattern()
        b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

        if not (_is_bullish(b1) and _is_bearish(b3)): return self._no_pattern()
        if not _is_doji(b2): return self._no_pattern()
        if float(b2["low"]) <= float(b1["high"]): return self._no_pattern()
        if float(b3["high"]) >= float(b2["low"]): return self._no_pattern()

        conf = min(1.0, 0.80 + self._vol_bonus(_f(b3, "relative_volume", 1.0)))
        return self._result(conf)


# ── Rising Three Methods ──────────────────────────────────────────────────────

class RisingThreeMethodsDetector(BasePatternDetector):
    """
    Rising Three Methods: large bullish | 3 small bearish inside | large bullish.
    Bullish continuation pattern.
    """
    min_bars_required = 5

    @property
    def name(self) -> str: return "rising_three_methods"

    @property
    def bias(self) -> PatternBias: return PatternBias.BULLISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 5: return self._no_pattern()
        b1, b2, b3, b4, b5 = [df.iloc[i] for i in [-5, -4, -3, -2, -1]]

        if not (_is_bullish(b1) and _is_bullish(b5)): return self._no_pattern()
        if _body_pct(b1) < self._threshold(0.60): return self._no_pattern()
        if _body_pct(b5) < self._threshold(0.60): return self._no_pattern()

        # Middle 3 bars should be bearish or small
        for b in [b2, b3, b4]:
            if _body_pct(b) > 0.50: return self._no_pattern()

        # Middle bars should be inside B1 range
        b1_range_ok = all(
            float(b["high"]) <= float(b1["high"]) and float(b["low"]) >= float(b1["low"])
            for b in [b2, b3, b4]
        )
        if not b1_range_ok: return self._no_pattern({"reason": "middle bars outside B1"})

        # B5 closes above B1 close
        if float(b5["close"]) <= float(b1["close"]): return self._no_pattern()

        conf = min(1.0, 0.65 + self._vol_bonus(_f(b5, "relative_volume", 1.0)))
        return self._result(conf)


# ── Falling Three Methods ─────────────────────────────────────────────────────

class FallingThreeMethodsDetector(BasePatternDetector):
    """Falling Three Methods: large bearish | 3 small bullish | large bearish."""
    min_bars_required = 5

    @property
    def name(self) -> str: return "falling_three_methods"

    @property
    def bias(self) -> PatternBias: return PatternBias.BEARISH

    def detect(self, df: pd.DataFrame) -> PatternResult:
        if len(df) < 5: return self._no_pattern()
        b1, b2, b3, b4, b5 = [df.iloc[i] for i in [-5, -4, -3, -2, -1]]

        if not (_is_bearish(b1) and _is_bearish(b5)): return self._no_pattern()
        if _body_pct(b1) < self._threshold(0.60): return self._no_pattern()
        if _body_pct(b5) < self._threshold(0.60): return self._no_pattern()

        for b in [b2, b3, b4]:
            if _body_pct(b) > 0.50: return self._no_pattern()

        b1_range_ok = all(
            float(b["high"]) <= float(b1["high"]) and float(b["low"]) >= float(b1["low"])
            for b in [b2, b3, b4]
        )
        if not b1_range_ok: return self._no_pattern()
        if float(b5["close"]) >= float(b1["close"]): return self._no_pattern()

        conf = min(1.0, 0.65 + self._vol_bonus(_f(b5, "relative_volume", 1.0)))
        return self._result(conf)

"""
Volume Context Engine.

Analyses volume behaviour on the latest bar relative to history and
returns a VolumeContext with a directional confirmation score that can
be consumed by the confluence engine.

Design rules:
  - All results are immutable (frozen dataclass)
  - No magic numbers — all thresholds are named class constants
  - Never raises on an empty / malformed DataFrame; returns safe defaults
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from libs.core.models.domain import SignalAction

# ── Volume trend literals ─────────────────────────────────────────────────────

_RISING = "rising"
_FALLING = "falling"
_NEUTRAL = "neutral"

# ── Effort-vs-result literals ─────────────────────────────────────────────────

_HIGH_EFFORT_LOW_RESULT = "high_effort_low_result"
_LOW_EFFORT_HIGH_RESULT = "low_effort_high_result"
_NORMAL = "normal"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VolumeContext:
    """Immutable snapshot of volume conditions for the latest bar."""

    relative_volume: float        # current bar vol / rolling-avg vol (1.0 = average)
    volume_trend: str             # "rising" | "falling" | "neutral"
    breakout_confirmed: bool      # high rel_vol + directional bar + rising trend
    reversal_confirmed: bool      # high rel_vol on a reversal bar
    divergence: bool              # price extreme on falling volume
    effort_vs_result: str         # "high_effort_low_result" | "low_effort_high_result" | "normal"
    above_vwap: bool              # latest close > VWAP (True when VWAP unavailable)
    score: float                  # 0.0–1.0 confirmation score for the proposed action


# ── Engine ────────────────────────────────────────────────────────────────────

class VolumeEngine:
    """Compute volume-based confirmation context for a proposed signal action."""

    # ── Thresholds ────────────────────────────────────────────────────────────
    HIGH_REL_VOL: float = 1.5
    LOW_REL_VOL: float = 0.6
    DIVERGENCE_LOOKBACK: int = 5

    # ── Scoring adjustments ───────────────────────────────────────────────────
    _SCORE_BASE: float = 0.5
    _SCORE_BREAKOUT: float = 0.20
    _SCORE_REVERSAL: float = 0.15
    _SCORE_TREND_MATCH: float = 0.10
    _SCORE_DIVERGENCE_PENALTY: float = 0.20
    _SCORE_HIGH_EFFORT_PENALTY: float = 0.15
    _SCORE_LOW_EFFORT_BONUS: float = 0.10

    # ── Effort-vs-result thresholds ───────────────────────────────────────────
    _LOW_BODY_PCT: float = 0.3
    _HIGH_BODY_PCT: float = 0.6

    # ── Divergence threshold ──────────────────────────────────────────────────
    _DIVERGENCE_VOL_RATIO: float = 0.8   # vol < 80 % of prior bar = "falling"

    def __init__(self, avg_period: int = 20) -> None:
        self._avg_period = avg_period

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        proposed_action: SignalAction,
    ) -> VolumeContext:
        """
        Return a VolumeContext for *proposed_action* based on *df*.

        Safe: never raises.  An empty / invalid DataFrame returns a
        neutral context with score = 0.5.
        """
        _default = VolumeContext(
            relative_volume=1.0,
            volume_trend=_NEUTRAL,
            breakout_confirmed=False,
            reversal_confirmed=False,
            divergence=False,
            effort_vs_result=_NORMAL,
            above_vwap=True,
            score=0.5,
        )

        if df is None or df.empty or len(df) < 2:
            return _default

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return _default

        try:
            rel_vol = self._relative_volume(df)
            vol_trend = self._volume_trend(df)
            divergence = self._divergence(df)
            effort_result = self._effort_vs_result(df, rel_vol)
            above_vwap = self._above_vwap(df)

            last_close = float(df["close"].iloc[-1])
            last_open = float(df["open"].iloc[-1])
            last_is_bullish = last_close > last_open
            last_is_bearish = last_close < last_open

            # Breakout: high vol + directional bar + rising trend
            breakout_confirmed = (
                rel_vol >= self.HIGH_REL_VOL
                and vol_trend == _RISING
                and (
                    (proposed_action == SignalAction.BUY and last_is_bullish)
                    or (proposed_action == SignalAction.SELL and last_is_bearish)
                )
            )

            # Reversal: high vol + last bar reverses prior direction
            reversal_confirmed = self._reversal_confirmed(
                df, rel_vol, proposed_action
            )

            score = self._score(
                rel_vol=rel_vol,
                vol_trend=vol_trend,
                breakout_confirmed=breakout_confirmed,
                reversal_confirmed=reversal_confirmed,
                divergence=divergence,
                effort_result=effort_result,
                proposed_action=proposed_action,
            )

            return VolumeContext(
                relative_volume=rel_vol,
                volume_trend=vol_trend,
                breakout_confirmed=breakout_confirmed,
                reversal_confirmed=reversal_confirmed,
                divergence=divergence,
                effort_vs_result=effort_result,
                above_vwap=above_vwap,
                score=score,
            )

        except Exception:  # noqa: BLE001 — broad catch, return safe default
            return _default

    # ── Private helpers ───────────────────────────────────────────────────────

    def _relative_volume(self, df: pd.DataFrame) -> float:
        """Current bar volume divided by rolling-average volume."""
        vol = df["volume"]
        avg = vol.rolling(self._avg_period).mean().iloc[-1]
        if pd.isna(avg) or avg == 0:
            return 1.0
        return float(vol.iloc[-1] / avg)

    def _volume_trend(self, df: pd.DataFrame, lookback: int = 5) -> str:
        """
        Fit a linear slope to the last *lookback* volume bars.

        Rising  : positive slope AND last bar > median of the window
        Falling : negative slope AND last bar < median of the window
        Neutral : everything else
        """
        window = df["volume"].iloc[-lookback:]
        if len(window) < 2:
            return _NEUTRAL

        x = np.arange(len(window), dtype=float)
        y = window.values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        median = float(np.median(y))
        last = float(y[-1])

        if slope > 0 and last > median:
            return _RISING
        if slope < 0 and last < median:
            return _FALLING
        return _NEUTRAL

    def _divergence(self, df: pd.DataFrame) -> bool:
        """
        Price divergence: price makes a new extreme but volume is falling.

        Bullish divergence (for bearish signals): new close high on lower vol.
        Bearish divergence (for bullish signals): new close low on lower vol.
        """
        window = df.iloc[-self.DIVERGENCE_LOOKBACK :]
        if len(window) < 2:
            return False

        closes = window["close"].values.astype(float)
        volumes = window["volume"].values.astype(float)

        last_close = closes[-1]
        last_vol = volumes[-1]
        prior_vol = volumes[-2]

        vol_falling = last_vol < prior_vol * self._DIVERGENCE_VOL_RATIO

        # New high on falling volume
        new_high = last_close == np.max(closes)
        # New low on falling volume
        new_low = last_close == np.min(closes)

        return bool((new_high or new_low) and vol_falling)

    def _effort_vs_result(self, df: pd.DataFrame, rel_vol: float) -> str:
        """
        Compare body percentage to relative volume to detect effort/result imbalance.

        Uses pre-computed columns if present; falls back to raw OHLC.
        """
        if "body_pct" in df.columns:
            body_pct = float(df["body_pct"].iloc[-1])
        else:
            open_ = float(df["open"].iloc[-1])
            high_ = float(df["high"].iloc[-1])
            low_ = float(df["low"].iloc[-1])
            close_ = float(df["close"].iloc[-1])
            total_range = high_ - low_
            body_size = abs(close_ - open_)
            body_pct = body_size / total_range if total_range > 0 else 0.0

        if pd.isna(body_pct):
            body_pct = 0.0

        if rel_vol > self.HIGH_REL_VOL and body_pct < self._LOW_BODY_PCT:
            return _HIGH_EFFORT_LOW_RESULT
        if rel_vol < self.LOW_REL_VOL and body_pct > self._HIGH_BODY_PCT:
            return _LOW_EFFORT_HIGH_RESULT
        return _NORMAL

    def _above_vwap(self, df: pd.DataFrame) -> bool:
        """True if latest close > latest VWAP.  Defaults to True when unavailable."""
        if "vwap" not in df.columns:
            return True
        vwap_val = df["vwap"].iloc[-1]
        if pd.isna(vwap_val):
            return True
        return bool(df["close"].iloc[-1] > vwap_val)

    def _reversal_confirmed(
        self,
        df: pd.DataFrame,
        rel_vol: float,
        proposed_action: SignalAction,
    ) -> bool:
        """
        High volume on a bar that reverses the prior bar's direction.

        BUY  reversal: last bar is bullish after a prior bearish bar.
        SELL reversal: last bar is bearish after a prior bullish bar.
        """
        if len(df) < 2:
            return False
        if rel_vol < self.HIGH_REL_VOL:
            return False

        last_close = float(df["close"].iloc[-1])
        last_open = float(df["open"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        prev_open = float(df["open"].iloc[-2])

        last_bullish = last_close > last_open
        last_bearish = last_close < last_open
        prev_bullish = prev_close > prev_open
        prev_bearish = prev_close < prev_open

        if proposed_action == SignalAction.BUY:
            return last_bullish and prev_bearish
        if proposed_action == SignalAction.SELL:
            return last_bearish and prev_bullish
        return False

    def _score(
        self,
        rel_vol: float,
        vol_trend: str,
        breakout_confirmed: bool,
        reversal_confirmed: bool,
        divergence: bool,
        effort_result: str,
        proposed_action: SignalAction,
    ) -> float:
        """
        Compute a [0.0, 1.0] confirmation score for the proposed action.

        NO_TRADE always returns 0.5 (neutral).
        """
        if proposed_action == SignalAction.NO_TRADE:
            return 0.5

        score = self._SCORE_BASE

        if breakout_confirmed:
            score += self._SCORE_BREAKOUT

        if reversal_confirmed:
            score += self._SCORE_REVERSAL

        # Trend alignment bonus
        if vol_trend == _RISING and proposed_action == SignalAction.BUY:
            score += self._SCORE_TREND_MATCH
        elif vol_trend == _FALLING and proposed_action == SignalAction.SELL:
            score += self._SCORE_TREND_MATCH

        if divergence:
            score -= self._SCORE_DIVERGENCE_PENALTY

        if effort_result == _HIGH_EFFORT_LOW_RESULT:
            score -= self._SCORE_HIGH_EFFORT_PENALTY
        elif effort_result == _LOW_EFFORT_HIGH_RESULT:
            score += self._SCORE_LOW_EFFORT_BONUS

        return float(np.clip(score, 0.0, 1.0))

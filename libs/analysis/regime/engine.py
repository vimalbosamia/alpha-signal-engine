"""
Regime / Volatility Engine.

Classifies the current market regime (trending, ranging, breakout, climactic)
using ATR-based volatility metrics and dual-EMA trend bias.  Returns an
immutable RegimeAnalysis with a vol_score usable by the confluence engine.

Design rules:
  - All results are immutable (frozen dataclass)
  - No magic numbers — all thresholds are named class constants
  - Never raises on an empty / malformed DataFrame; returns safe defaults
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from libs.core.models.domain import MarketRegime


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeAnalysis:
    """Immutable snapshot of volatility and regime conditions."""

    regime: MarketRegime
    atr: float                        # ATR value in price units
    atr_pct: float                    # ATR as % of current price (atr / close * 100)
    is_high_vol: bool                 # atr_pct above high-vol threshold
    is_expanding: bool                # ATR growing (last > prior by >10%)
    is_contracting: bool              # ATR shrinking (last < prior by >10%)
    vol_score: float                  # 0.0–1.0 trade-appropriateness score
    ema_fast: float | None = None     # last value of fast EMA
    ema_slow: float | None = None     # last value of slow EMA
    notes: str = ""


# ── Engine ────────────────────────────────────────────────────────────────────

class RegimeEngine:
    """Classify market regime and assess volatility suitability for trading."""

    # ── Period constants ──────────────────────────────────────────────────────
    ATR_PERIOD: int = 14
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    VOL_LOOKBACK: int = 50

    # ── Volatility thresholds ─────────────────────────────────────────────────
    HIGH_VOL_MULTIPLIER: float = 1.5   # atr_pct > median * multiplier → high vol
    CLIMACTIC_MULTIPLIER: float = 3.0  # extreme spike → climactic
    EXPANSION_THRESHOLD: float = 0.10  # 10% ATR growth = expanding

    # ── EMA trend threshold ───────────────────────────────────────────────────
    _EMA_TREND_THRESHOLD: float = 0.001  # 0.1% difference to call a trend

    # ── Minimum bars required ─────────────────────────────────────────────────
    _MIN_BARS: int = 3

    # ── Default vol_score when data is insufficient ───────────────────────────
    _DEFAULT_VOL_SCORE: float = 0.5

    def __init__(
        self,
        atr_period: int = ATR_PERIOD,
        ema_fast: int = EMA_FAST,
        ema_slow: int = EMA_SLOW,
        vol_lookback: int = VOL_LOOKBACK,
    ) -> None:
        self._atr_period = atr_period
        self._ema_fast_span = ema_fast
        self._ema_slow_span = ema_slow
        self._vol_lookback = vol_lookback

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> RegimeAnalysis:
        """
        Return a RegimeAnalysis for the latest bar in *df*.

        Safe: never raises.  An empty / short DataFrame returns a safe
        UNKNOWN regime with vol_score = 0.5.
        """
        _unknown = RegimeAnalysis(
            regime=MarketRegime.UNKNOWN,
            atr=0.0,
            atr_pct=0.0,
            is_high_vol=False,
            is_expanding=False,
            is_contracting=False,
            vol_score=self._DEFAULT_VOL_SCORE,
        )

        if df is None or df.empty or len(df) < self._MIN_BARS:
            return _unknown

        required = {"open", "high", "low", "close"}
        if not required.issubset(df.columns):
            return _unknown

        try:
            atr_series = self._calc_atr(df)
            ema_fast_series = self._calc_ema(df["close"], self._ema_fast_span)
            ema_slow_series = self._calc_ema(df["close"], self._ema_slow_span)

            last_close = float(df["close"].iloc[-1])
            if last_close <= 0:
                return _unknown

            last_atr = float(atr_series.iloc[-1])
            atr_pct = last_atr / last_close * 100.0

            # Determine ATR expansion / contraction using last two ATR values
            is_expanding = False
            is_contracting = False
            if len(atr_series) >= 2:
                prior_atr = float(atr_series.iloc[-2])
                if prior_atr > 0:
                    change = (last_atr - prior_atr) / prior_atr
                    if change > self.EXPANSION_THRESHOLD:
                        is_expanding = True
                    elif change < -self.EXPANSION_THRESHOLD:
                        is_contracting = True

            # Compute median atr_pct over lookback window for vol classification
            atr_pct_series = atr_series / df["close"] * 100.0
            lookback_window = atr_pct_series.iloc[-self._vol_lookback:]
            median_atr_pct = float(np.median(lookback_window.dropna().values)) if len(lookback_window) > 0 else atr_pct

            is_high_vol = atr_pct > median_atr_pct * self.HIGH_VOL_MULTIPLIER

            last_ema_fast = float(ema_fast_series.iloc[-1])
            last_ema_slow = float(ema_slow_series.iloc[-1])

            regime = self._classify_regime(
                df=df,
                atr_series=atr_pct_series,
                ema_fast_series=ema_fast_series,
                ema_slow_series=ema_slow_series,
            )

            vol_score = self._vol_score(
                atr_pct=atr_pct,
                is_high_vol=is_high_vol,
                is_expanding=is_expanding,
            )

            # Apply climactic penalty inside vol_score path
            if regime == MarketRegime.CLIMACTIC:
                vol_score = float(np.clip(vol_score - 0.3, 0.0, 1.0))

            return RegimeAnalysis(
                regime=regime,
                atr=last_atr,
                atr_pct=atr_pct,
                is_high_vol=is_high_vol,
                is_expanding=is_expanding,
                is_contracting=is_contracting,
                vol_score=vol_score,
                ema_fast=last_ema_fast,
                ema_slow=last_ema_slow,
            )

        except Exception:  # noqa: BLE001 — broad catch, return safe default
            return _unknown

    # ── Private helpers ───────────────────────────────────────────────────────

    def _calc_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        True Average True Range using Wilder smoothing (ewm, adjust=False).

        Returns pd.Series([0.0]) if df has fewer than 2 rows.
        """
        if len(df) < 2:
            return pd.Series([0.0], index=df.index[:1] if not df.empty else None)

        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.ewm(span=self._atr_period, adjust=False).mean()
        return atr

    def _calc_ema(self, series: pd.Series, span: int) -> pd.Series:
        """
        Exponential moving average using ewm(span, adjust=False).

        Returns a series of zeros on empty input.
        """
        if series.empty:
            return pd.Series(dtype=float)
        return series.ewm(span=span, adjust=False).mean()

    def _classify_regime(
        self,
        df: pd.DataFrame,
        atr_series: pd.Series,
        ema_fast_series: pd.Series,
        ema_slow_series: pd.Series,
    ) -> MarketRegime:
        """
        Classify the current regime.  Evaluation order matters — first match wins.

        1. Climactic  — extreme ATR spike
        2. Trending up (high vol + fast > slow)
        3. Trending down (high vol + fast < slow)
        4. Ranging high vol
        5. Breakout (ATR expanding)
        6. Trending up (low vol, EMA crossover)
        7. Trending down (low vol, EMA crossover)
        8. Ranging low vol (default)
        """
        last_atr_pct = float(atr_series.iloc[-1])
        lookback_window = atr_series.iloc[-self._vol_lookback:]
        median_atr_pct = float(
            np.median(lookback_window.dropna().values)
        ) if len(lookback_window) > 0 else last_atr_pct

        last_ema_fast = float(ema_fast_series.iloc[-1])
        last_ema_slow = float(ema_slow_series.iloc[-1])

        # Avoid division by zero for EMA comparison
        ema_diff_pct = (
            (last_ema_fast - last_ema_slow) / last_ema_slow
            if last_ema_slow != 0
            else 0.0
        )

        # 1. Climactic
        if median_atr_pct > 0 and last_atr_pct > median_atr_pct * self.CLIMACTIC_MULTIPLIER:
            return MarketRegime.CLIMACTIC

        # 2–4. High volatility regimes
        if median_atr_pct > 0 and last_atr_pct > median_atr_pct * self.HIGH_VOL_MULTIPLIER:
            if last_ema_fast > last_ema_slow:
                return MarketRegime.TRENDING_UP
            if last_ema_fast < last_ema_slow:
                return MarketRegime.TRENDING_DOWN
            return MarketRegime.RANGING_HIGH_VOL

        # 5. Breakout (expanding ATR, not climactic)
        if len(atr_series) >= 2:
            prior_atr_pct = float(atr_series.iloc[-2])
            if prior_atr_pct > 0:
                atr_change = (last_atr_pct - prior_atr_pct) / prior_atr_pct
                if atr_change > self.EXPANSION_THRESHOLD:
                    return MarketRegime.BREAKOUT

        # 6. Trending up (EMA crossover, low vol)
        if ema_diff_pct > self._EMA_TREND_THRESHOLD:
            return MarketRegime.TRENDING_UP

        # 7. Trending down (EMA crossover, low vol)
        if ema_diff_pct < -self._EMA_TREND_THRESHOLD:
            return MarketRegime.TRENDING_DOWN

        # 8. Default — quiet ranging market
        return MarketRegime.RANGING_LOW_VOL

    def _vol_score(
        self,
        atr_pct: float,
        is_high_vol: bool,
        is_expanding: bool,
    ) -> float:
        """
        Compute a [0.0, 1.0] score for how trade-appropriate the volatility is.

        Base: 0.5
        +0.3  low-vol range (controllable, low slippage)
        +0.2  expanding ATR (breakout opportunity)
        -0.2  high-vol, not expanding (uncertain noise)
        -0.3  applied externally for climactic (dangerous spike)
        """
        score = 0.5

        if not is_high_vol and not is_expanding:
            score += 0.3
        if is_expanding:
            score += 0.2
        if is_high_vol and not is_expanding:
            score -= 0.2

        return float(np.clip(score, 0.0, 1.0))

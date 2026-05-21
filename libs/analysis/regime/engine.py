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

         1. Panic selloff   — extreme drop + extreme volume
         2. Liquidation     — climactic vol + large consecutive candles
         3. Climactic       — extreme ATR spike
         4. News-driven     — sudden vol spike from low base
         5. Compression     — ATR contracting below 50% of median
         6. Expansion       — ATR expanding 2x+ from compressed state
         7. Reversal        — trend reversal signal (EMA crossover flip)
         8. Accumulation    — ranging after downtrend with rising volume
         9. Distribution    — ranging after uptrend with rising volume
        10. Low liquidity   — very low volume vs average
        11. Mean reversion  — extreme RSI + EMA convergence
        12. Trending up     — high vol + fast > slow
        13. Trending down   — high vol + fast < slow
        14. Ranging high vol
        15. Breakout        — ATR expanding
        16. Trending up     — low vol, EMA crossover
        17. Trending down   — low vol, EMA crossover
        18. Ranging low vol — default
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

        # Volume analysis (best-effort — volume column may not exist)
        has_volume = "volume" in df.columns and len(df) >= self._vol_lookback
        volume_ratio = 1.0
        if has_volume:
            avg_vol = float(df["volume"].iloc[-self._vol_lookback:].mean())
            last_vol = float(df["volume"].iloc[-1])
            volume_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        # Price change analysis
        last_close = float(df["close"].iloc[-1])
        price_change_pct = 0.0
        if len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            price_change_pct = (
                (last_close - prev_close) / prev_close * 100
                if prev_close > 0 else 0.0
            )

        # Prior trend for accumulation/distribution (20-bar lookback)
        prior_trend = "flat"
        if len(ema_fast_series) >= 20:
            ema_20_ago = float(ema_fast_series.iloc[-20])
            if ema_20_ago > 0:
                trend_change = (last_ema_fast - ema_20_ago) / ema_20_ago
                if trend_change > 0.02:
                    prior_trend = "up"
                elif trend_change < -0.02:
                    prior_trend = "down"

        # EMA crossover flip detection (reversal)
        ema_crossed = False
        if len(ema_fast_series) >= 3 and len(ema_slow_series) >= 3:
            prev_fast = float(ema_fast_series.iloc[-3])
            prev_slow = float(ema_slow_series.iloc[-3])
            if prev_slow != 0 and last_ema_slow != 0:
                prev_diff = (prev_fast - prev_slow) / prev_slow
                curr_diff = ema_diff_pct
                if (prev_diff > self._EMA_TREND_THRESHOLD and curr_diff < -self._EMA_TREND_THRESHOLD) or \
                   (prev_diff < -self._EMA_TREND_THRESHOLD and curr_diff > self._EMA_TREND_THRESHOLD):
                    ema_crossed = True

        # ATR history for compression/expansion detection
        prior_median_atr = median_atr_pct
        if len(atr_series) >= self._vol_lookback + 10:
            prior_window = atr_series.iloc[-(self._vol_lookback + 10):-10]
            if len(prior_window) > 0:
                prior_median_atr = float(np.median(prior_window.dropna().values))

        # ── Classification cascade ───────────────────────────────────────────

        # 1. Panic selloff — extreme drop + high volume
        if (price_change_pct < -5.0 and volume_ratio > 2.0 and
                median_atr_pct > 0 and last_atr_pct > median_atr_pct * 2.0):
            return MarketRegime.PANIC_SELLOFF

        # 2. Liquidation event — extreme vol + consecutive large candles
        if median_atr_pct > 0 and last_atr_pct > median_atr_pct * self.CLIMACTIC_MULTIPLIER:
            if len(df) >= 3:
                recent_ranges = [
                    abs(float(df["close"].iloc[i]) - float(df["open"].iloc[i]))
                    for i in range(-3, 0)
                ]
                avg_body = sum(recent_ranges) / 3
                if avg_body > 0 and last_close > 0 and (avg_body / last_close * 100) > 2.0:
                    return MarketRegime.LIQUIDATION_EVENT

        # 3. Climactic — extreme ATR spike
        if median_atr_pct > 0 and last_atr_pct > median_atr_pct * self.CLIMACTIC_MULTIPLIER:
            return MarketRegime.CLIMACTIC

        # 4. News-driven — sudden vol spike from previously low base
        if (prior_median_atr > 0 and
                last_atr_pct > prior_median_atr * 2.5 and
                prior_median_atr < median_atr_pct * 0.7):
            return MarketRegime.NEWS_DRIVEN

        # 5. Compression — ATR well below median
        if median_atr_pct > 0 and last_atr_pct < median_atr_pct * 0.5:
            return MarketRegime.COMPRESSION

        # 6. Expansion — ATR expanding from compressed state
        if len(atr_series) >= 5:
            recent_atr_min = float(atr_series.iloc[-5:].min())
            if (median_atr_pct > 0 and
                    recent_atr_min < median_atr_pct * 0.6 and
                    last_atr_pct > median_atr_pct * 1.2):
                return MarketRegime.EXPANSION

        # 7. Reversal — EMA crossover flip
        if ema_crossed and abs(ema_diff_pct) < 0.01:
            return MarketRegime.REVERSAL

        # 8. Accumulation — ranging after downtrend, volume rising
        if (prior_trend == "down" and
                abs(ema_diff_pct) < self._EMA_TREND_THRESHOLD * 2 and
                volume_ratio > 1.3):
            return MarketRegime.ACCUMULATION

        # 9. Distribution — ranging after uptrend, volume rising
        if (prior_trend == "up" and
                abs(ema_diff_pct) < self._EMA_TREND_THRESHOLD * 2 and
                volume_ratio > 1.3):
            return MarketRegime.DISTRIBUTION

        # 10. Low liquidity — very low volume
        if has_volume and volume_ratio < 0.3:
            return MarketRegime.LOW_LIQUIDITY

        # 11. Mean reversion — compute simple RSI proxy
        if len(df) >= 14:
            closes = df["close"].iloc[-14:].values
            deltas = np.diff(closes)
            gains = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0.0
            losses = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 0.0
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0 if gains > 0 else 50.0

            if (rsi < 25 or rsi > 75) and abs(ema_diff_pct) < self._EMA_TREND_THRESHOLD * 3:
                return MarketRegime.MEAN_REVERSION

        # 12–14. High volatility regimes
        if median_atr_pct > 0 and last_atr_pct > median_atr_pct * self.HIGH_VOL_MULTIPLIER:
            if last_ema_fast > last_ema_slow:
                return MarketRegime.TRENDING_UP
            if last_ema_fast < last_ema_slow:
                return MarketRegime.TRENDING_DOWN
            return MarketRegime.RANGING_HIGH_VOL

        # 15. Breakout (expanding ATR, not climactic)
        if len(atr_series) >= 2:
            prior_atr_pct = float(atr_series.iloc[-2])
            if prior_atr_pct > 0:
                atr_change = (last_atr_pct - prior_atr_pct) / prior_atr_pct
                if atr_change > self.EXPANSION_THRESHOLD:
                    return MarketRegime.BREAKOUT

        # 16. Trending up (EMA crossover, low vol)
        if ema_diff_pct > self._EMA_TREND_THRESHOLD:
            return MarketRegime.TRENDING_UP

        # 17. Trending down (EMA crossover, low vol)
        if ema_diff_pct < -self._EMA_TREND_THRESHOLD:
            return MarketRegime.TRENDING_DOWN

        # 18. Default — quiet ranging market
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

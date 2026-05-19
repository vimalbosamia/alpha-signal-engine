"""
Momentum Continuation Strategy.

Logic: In a trend (ADX > 25), after a small pullback (RSI dipped to 40-50 in
uptrend, or 50-60 in downtrend), momentum resumes when RSI crosses back and
MACD histogram is increasing.

Design rules:
  - Never raises in generate_candidate — returns None on any data issue.
  - No execution logic — produces signal candidates only.
  - All thresholds are named constants.
"""
from __future__ import annotations

import pandas as pd

from libs.core.models.domain import (
    AssetClass,
    DataQualityReport,
    MarketRegime,
    SessionState,
    SignalAction,
    SignalCandidate,
    Timeframe,
    TrendDirection,
)
from libs.analysis.indicators.engine import IndicatorSnapshot
from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.analysis.volume.engine import VolumeContext
from libs.strategies.base.strategy import BaseStrategy


class MomentumContinuationStrategy(BaseStrategy):
    """
    Momentum continuation after a brief RSI pullback in a trending market.

    BUY:  UPTREND + ADX > 25 + RSI dipped below 50 in last 3 bars + RSI now > 50
          + MACD histogram increasing
    SELL: DOWNTREND + ADX > 25 + RSI rose above 50 in last 3 bars + RSI now < 50
          + MACD histogram decreasing (more negative)
    """

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "momentum_continuation"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 20

    # ── Thresholds ─────────────────────────────────────────────────────────────

    MIN_ADX: float = 25.0
    RSI_MID: float = 50.0
    RSI_PULLBACK_LOOKBACK: int = 3      # bars to look back for RSI dip
    ATR_STOP_MULTIPLIER: float = 2.0
    ATR_TP_MULTIPLIER: float = 4.0
    CONFIDENCE: float = 0.62

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_candidate(
        self,
        symbol: str,
        asset_class: AssetClass,
        df: pd.DataFrame,
        df_htf: pd.DataFrame | None,
        session: SessionState | None,
        quality: DataQualityReport | None,
        structure: MarketStructure | None,
        levels: list[KeyLevel] | None,
        volume: VolumeContext | None,
        regime: RegimeAnalysis | None,
        indicators: IndicatorSnapshot | None = None,
    ) -> SignalCandidate | None:
        try:
            # 1. Pre-flight eligibility
            if not self.is_eligible(asset_class, session, quality):
                return None

            # 2. Require minimum bars
            if df is None or len(df) < self.min_bars_required:
                return None

            # 3. Require indicator data
            if indicators is None:
                return None

            adx = indicators.adx
            rsi = indicators.rsi
            macd_histogram = indicators.macd_histogram
            macd_histogram_prev = indicators.macd_histogram_prev

            if adx is None or rsi is None:
                return None

            # 4. Require strong trend (ADX > 25)
            if adx < self.MIN_ADX:
                return None

            # 5. Require trend direction
            if structure is None:
                return None
            trend = structure.trend

            last_close = float(df["close"].iloc[-1])
            atr = indicators.atr
            if atr is None or atr <= 0.0:
                return None

            # 6. Direction-specific setup check
            if trend == TrendDirection.UPTREND:
                action = self._check_buy_setup(df, rsi, macd_histogram, macd_histogram_prev)
                if action is None:
                    return None
                proposed_action = SignalAction.BUY
                stop_loss = last_close - self.ATR_STOP_MULTIPLIER * atr
                take_profit_1 = last_close + self.ATR_TP_MULTIPLIER * atr

            elif trend == TrendDirection.DOWNTREND:
                action = self._check_sell_setup(df, rsi, macd_histogram, macd_histogram_prev)
                if action is None:
                    return None
                proposed_action = SignalAction.SELL
                stop_loss = last_close + self.ATR_STOP_MULTIPLIER * atr
                take_profit_1 = last_close - self.ATR_TP_MULTIPLIER * atr

            else:
                return None

            # 7. Validate stop / take-profit geometry
            if stop_loss <= 0.0:
                return None
            if proposed_action == SignalAction.BUY and stop_loss >= last_close:
                return None
            if proposed_action == SignalAction.SELL and stop_loss <= last_close:
                return None

            # 8. Entry zone
            entry_low = last_close * 0.9998
            entry_high = last_close * 1.0002

            higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN

            return SignalCandidate(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=self.name,
                proposed_action=proposed_action,
                timeframe=Timeframe.FIFTEEN_MIN,
                higher_tf_bias=higher_tf_bias,
                entry_zone_low=entry_low,
                entry_zone_high=entry_high,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                session=session,
                quality=quality,
                raw_features={
                    "adx": adx,
                    "rsi": rsi,
                    "macd_histogram": macd_histogram,
                    "macd_histogram_prev": macd_histogram_prev,
                    "atr": atr,
                    "confidence": self.CONFIDENCE,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_buy_setup(
        self,
        df: pd.DataFrame,
        rsi: float,
        macd_histogram: float | None,
        macd_histogram_prev: float | None,
    ) -> bool | None:
        """
        Return True if BUY setup is valid:
          - RSI was below 50 in the last RSI_PULLBACK_LOOKBACK bars
          - Current RSI is above 50 (momentum resuming)
          - MACD histogram is increasing (or MACD data unavailable — be lenient)
        """
        # Current RSI must be above midline
        if rsi <= self.RSI_MID:
            return None

        # RSI must have dipped below 50 recently (pullback occurred)
        if not self._rsi_dipped_below(df, self.RSI_MID, self.RSI_PULLBACK_LOOKBACK):
            return None

        # MACD histogram must be increasing (current > prev)
        if macd_histogram is not None and macd_histogram_prev is not None:
            if macd_histogram <= macd_histogram_prev:
                return None

        return True

    def _check_sell_setup(
        self,
        df: pd.DataFrame,
        rsi: float,
        macd_histogram: float | None,
        macd_histogram_prev: float | None,
    ) -> bool | None:
        """
        Return True if SELL setup is valid:
          - RSI was above 50 in the last RSI_PULLBACK_LOOKBACK bars (counter-trend bounce)
          - Current RSI is below 50 (momentum resuming downward)
          - MACD histogram is decreasing (becoming more negative)
        """
        # Current RSI must be below midline
        if rsi >= self.RSI_MID:
            return None

        # RSI must have been above 50 recently (pullback bounce occurred)
        if not self._rsi_rose_above(df, self.RSI_MID, self.RSI_PULLBACK_LOOKBACK):
            return None

        # MACD histogram must be decreasing (current < prev, going more negative)
        if macd_histogram is not None and macd_histogram_prev is not None:
            if macd_histogram >= macd_histogram_prev:
                return None

        return True

    def _rsi_dipped_below(
        self,
        df: pd.DataFrame,
        threshold: float,
        lookback: int,
    ) -> bool:
        """
        Check if the 'rsi' column in df dipped below `threshold` in the
        last `lookback` bars (excluding the current bar).
        Falls back gracefully if the column is absent.
        """
        if "rsi" not in df.columns:
            return False
        n = len(df)
        start = max(0, n - lookback - 1)
        end = n - 1  # exclude current bar
        window = df["rsi"].iloc[start:end].dropna()
        return bool((window < threshold).any())

    def _rsi_rose_above(
        self,
        df: pd.DataFrame,
        threshold: float,
        lookback: int,
    ) -> bool:
        """
        Check if the 'rsi' column in df rose above `threshold` in the
        last `lookback` bars (excluding the current bar).
        Falls back gracefully if the column is absent.
        """
        if "rsi" not in df.columns:
            return False
        n = len(df)
        start = max(0, n - lookback - 1)
        end = n - 1  # exclude current bar
        window = df["rsi"].iloc[start:end].dropna()
        return bool((window > threshold).any())

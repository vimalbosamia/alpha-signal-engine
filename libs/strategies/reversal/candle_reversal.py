"""
Candlestick Reversal Multi-Pattern Strategy.

Logic: Detects bullish or bearish engulfing patterns from the last 2 bars.
  - BUY: last bar is a strong bullish engulfing (close > open, body > 70% of
    range, and the prior bar was bearish)
  - SELL: last bar is a strong bearish engulfing (close < open, body > 70% of
    range, and the prior bar was bullish)

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
    PatternBias,
    PatternResult,
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


class CandlestickReversalStrategy(BaseStrategy):
    """
    Multi-pattern candlestick reversal strategy.

    Triggers on a strong bullish or bearish engulfing pattern where the
    candle body occupies at least MIN_BODY_RANGE_PCT of the total range.
    """

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "candlestick_reversal"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 10

    # ── Thresholds ─────────────────────────────────────────────────────────────

    CONFIDENCE: float = 0.55
    ATR_STOP_MULTIPLIER: float = 1.0   # stop sits just beyond the pattern low/high
    TP_ATR_MULTIPLIER: float = 3.0     # TP = entry + ATR * 3

    # Engulfing: body must cover at least this fraction of bar's total range
    MIN_BODY_RANGE_PCT: float = 0.70

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
            if not self.is_eligible(asset_class, session, quality):
                return None

            if df is None or len(df) < self.min_bars_required:
                return None

            action = self._detect_signal(df)
            if action is None:
                return None

            last = df.iloc[-1]
            last_close = float(last["close"])
            last_low = float(last["low"])
            last_high = float(last["high"])

            atr_dist = self._atr_stop(df, 1.0)
            if atr_dist <= 0.0:
                return None

            if action == SignalAction.BUY:
                pattern_low = min(float(df.iloc[-2]["low"]), last_low)
                stop_loss = pattern_low - atr_dist * self.ATR_STOP_MULTIPLIER
                if stop_loss <= 0.0:
                    return None

                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                if stop_loss >= entry_mid:
                    return None

                take_profit_1 = entry_mid + atr_dist * self.TP_ATR_MULTIPLIER

                pattern_result = PatternResult(
                    pattern_name="bullish_engulfing",
                    detected=True,
                    confidence=self.CONFIDENCE,
                    bias=PatternBias.BULLISH,
                    candle_span=2,
                )

            else:  # SELL
                pattern_high = max(float(df.iloc[-2]["high"]), last_high)
                stop_loss = pattern_high + atr_dist * self.ATR_STOP_MULTIPLIER

                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                if stop_loss <= entry_mid:
                    return None

                take_profit_1 = entry_mid - atr_dist * self.TP_ATR_MULTIPLIER
                if take_profit_1 <= 0.0:
                    return None

                pattern_result = PatternResult(
                    pattern_name="bearish_engulfing",
                    detected=True,
                    confidence=self.CONFIDENCE,
                    bias=PatternBias.BEARISH,
                    candle_span=2,
                )

            higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN

            return SignalCandidate(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=self.name,
                proposed_action=action,
                timeframe=Timeframe.FIFTEEN_MIN,
                higher_tf_bias=higher_tf_bias,
                entry_zone_low=entry_low,
                entry_zone_high=entry_high,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                pattern_results=[pattern_result],
                regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                session=session,
                quality=quality,
                raw_features={
                    "last_close": last_close,
                    "atr_dist": atr_dist,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _detect_signal(self, df: pd.DataFrame) -> SignalAction | None:
        """
        Return BUY on a strong bullish engulfing, SELL on a strong bearish
        engulfing, or None if neither condition is met.

        Bullish engulfing:
          - Current bar is bullish (close > open)
          - Body covers >= MIN_BODY_RANGE_PCT of total range
          - Previous bar was bearish

        Bearish engulfing:
          - Current bar is bearish (close < open)
          - Body covers >= MIN_BODY_RANGE_PCT of total range
          - Previous bar was bullish
        """
        if len(df) < 2:
            return None

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        curr_open = float(curr["open"])
        curr_close = float(curr["close"])
        curr_high = float(curr["high"])
        curr_low = float(curr["low"])

        curr_range = curr_high - curr_low
        if curr_range <= 0.0:
            return None

        curr_body = abs(curr_close - curr_open)
        body_pct = curr_body / curr_range

        if body_pct < self.MIN_BODY_RANGE_PCT:
            return None

        prev_open = float(prev["open"])
        prev_close = float(prev["close"])

        prev_is_bearish = prev_close < prev_open
        prev_is_bullish = prev_close > prev_open

        if curr_close > curr_open and prev_is_bearish:
            return SignalAction.BUY

        if curr_close < curr_open and prev_is_bullish:
            return SignalAction.SELL

        return None

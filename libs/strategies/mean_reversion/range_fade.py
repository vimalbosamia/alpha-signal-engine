"""
Range Fade Strategy.

Logic: In a ranging market (ADX < 20 or regime is RANGING_*), price near the
top of the 20-bar range is overbought → SELL.  Price near the bottom is
oversold → BUY.  RSI confirmation is required to avoid fading into a breakout.

Design rules:
  - Never raises in generate_candidate — returns None on any data issue.
  - No execution logic — produces signal candidates only.
  - All thresholds are named constants.
"""
from __future__ import annotations

import pandas as pd

from libs.analysis.indicators.engine import IndicatorSnapshot
from libs.analysis.levels.engine import KeyLevel
from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure
from libs.analysis.volume.engine import VolumeContext
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
from libs.strategies.base.strategy import BaseStrategy


class RangeFadeStrategy(BaseStrategy):
    """Mean-reversion strategy that fades price extremes within a trading range."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "range_fade"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

    # ── Thresholds ─────────────────────────────────────────────────────────────

    RANGE_LOOKBACK: int = 20            # bars used to define the range
    EDGE_ZONE_PCT: float = 0.20         # top/bottom 20 % of range = trigger zone
    ADX_RANGE_THRESHOLD: float = 20.0   # ADX < this → ranging
    RSI_OVERBOUGHT: float = 65.0        # RSI confirmation for SELL at range top
    RSI_OVERSOLD: float = 35.0          # RSI confirmation for BUY at range bottom
    CONFIDENCE: float = 0.55
    ATR_STOP_MULTIPLIER: float = 2.0

    _RANGING_REGIMES = {
        MarketRegime.RANGING_LOW_VOL,
        MarketRegime.RANGING_HIGH_VOL,
    }

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

            # 3. Confirm ranging market via ADX or regime
            adx = indicators.adx if indicators is not None else None
            current_regime = regime.regime if regime is not None else MarketRegime.UNKNOWN

            is_ranging_by_regime = current_regime in self._RANGING_REGIMES
            is_ranging_by_adx = adx is not None and adx < self.ADX_RANGE_THRESHOLD

            if not (is_ranging_by_regime or is_ranging_by_adx):
                return None

            # 4. Compute 20-bar range
            lookback = df.iloc[-self.RANGE_LOOKBACK :]
            range_high = float(lookback["high"].max())
            range_low = float(lookback["low"].min())
            range_width = range_high - range_low

            if range_width <= 0.0:
                return None

            last_close = float(df["close"].iloc[-1])

            # 5. Determine which zone price is in
            top_zone_threshold = range_high - range_width * self.EDGE_ZONE_PCT
            bottom_zone_threshold = range_low + range_width * self.EDGE_ZONE_PCT

            # 6. Resolve RSI (prefer indicator snapshot, fall back to None)
            rsi = indicators.rsi if indicators is not None else None

            atr_dist = self._atr_stop(df, self.ATR_STOP_MULTIPLIER)
            if atr_dist <= 0.0:
                return None

            # 7. SELL: price near range top + RSI overbought confirmation
            if last_close >= top_zone_threshold:
                if rsi is not None and rsi < self.RSI_OVERBOUGHT:
                    return None  # no overbought confirmation

                stop_loss = range_high + atr_dist  # beyond range boundary
                take_profit_1 = range_low           # opposite side of range
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                if stop_loss <= entry_mid:
                    return None
                if take_profit_1 >= entry_mid:
                    return None
                if take_profit_1 <= 0.0:
                    return None

                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.SELL,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=self._htf_bias(structure),
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=current_regime,
                    session=session,
                    quality=quality,
                    raw_features={
                        "range_high": range_high,
                        "range_low": range_low,
                        "range_width": range_width,
                        "last_close": last_close,
                        "rsi": rsi,
                        "adx": adx,
                        "zone": "top",
                        "confidence": self.CONFIDENCE,
                    },
                )

            # 8. BUY: price near range bottom + RSI oversold confirmation
            if last_close <= bottom_zone_threshold:
                if rsi is not None and rsi > self.RSI_OVERSOLD:
                    return None  # no oversold confirmation

                stop_loss = range_low - atr_dist  # beyond range boundary
                if stop_loss <= 0.0:
                    return None
                take_profit_1 = range_high          # opposite side of range
                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                if stop_loss >= entry_mid:
                    return None
                if take_profit_1 <= entry_mid:
                    return None

                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.BUY,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=self._htf_bias(structure),
                    entry_zone_low=entry_low,
                    entry_zone_high=entry_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=current_regime,
                    session=session,
                    quality=quality,
                    raw_features={
                        "range_high": range_high,
                        "range_low": range_low,
                        "range_width": range_width,
                        "last_close": last_close,
                        "rsi": rsi,
                        "adx": adx,
                        "zone": "bottom",
                        "confidence": self.CONFIDENCE,
                    },
                )

        except Exception:  # noqa: BLE001 — never propagate
            return None

        return None

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _htf_bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure is not None else TrendDirection.UNKNOWN

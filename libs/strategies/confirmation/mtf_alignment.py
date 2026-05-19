"""
Multi-Timeframe Alignment Strategy.

Logic: Only triggers when BOTH the primary timeframe and higher timeframe
agree on direction.
  - BUY:  primary EMA-9 > EMA-20  AND  HTF last close > HTF 20-bar SMA
  - SELL: primary EMA-9 < EMA-20  AND  HTF last close < HTF 20-bar SMA
  - If df_htf is None → return None (can't confirm alignment)

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


class MTFAlignmentStrategy(BaseStrategy):
    """
    Multi-timeframe alignment strategy.

    Produces high-confidence signals only when both the primary timeframe
    EMA trend and the higher-timeframe SMA trend point the same direction.
    """

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "mtf_alignment"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

    # ── Thresholds ─────────────────────────────────────────────────────────────

    CONFIDENCE: float = 0.70
    ATR_STOP_MULTIPLIER: float = 2.0
    TP_ATR_MULTIPLIER: float = 4.0

    HTF_SMA_PERIOD: int = 20   # period for HTF SMA trend check

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

            # HTF is mandatory for alignment confirmation
            if df_htf is None or len(df_htf) < self.HTF_SMA_PERIOD:
                return None

            # ── Primary timeframe EMA direction ───────────────────────────────
            ema_9 = indicators.ema_9 if indicators is not None else None
            ema_20 = indicators.ema_20 if indicators is not None else None

            if ema_9 is None or ema_20 is None:
                return None

            primary_bullish = ema_9 > ema_20
            primary_bearish = ema_9 < ema_20

            # Neither condition is unambiguous — skip
            if not primary_bullish and not primary_bearish:
                return None

            # ── HTF trend check: last close vs. 20-bar SMA ────────────────────
            htf_close = float(df_htf["close"].iloc[-1])
            htf_sma = float(df_htf["close"].rolling(self.HTF_SMA_PERIOD).mean().iloc[-1])

            if pd.isna(htf_sma):
                return None

            htf_bullish = htf_close > htf_sma
            htf_bearish = htf_close < htf_sma

            # ── Require both timeframes to agree ─────────────────────────────
            if primary_bullish and htf_bullish:
                action = SignalAction.BUY
            elif primary_bearish and htf_bearish:
                action = SignalAction.SELL
            else:
                return None  # conflicting timeframes

            # ── Build trade levels ────────────────────────────────────────────
            last_close = float(df["close"].iloc[-1])
            atr_dist = self._atr_stop(df, 1.0)
            if atr_dist <= 0.0:
                return None

            entry_low = last_close * 0.9995
            entry_high = last_close * 1.0005
            entry_mid = (entry_low + entry_high) / 2.0

            if action == SignalAction.BUY:
                stop_loss = entry_mid - atr_dist * self.ATR_STOP_MULTIPLIER
                if stop_loss <= 0.0:
                    return None
                take_profit_1 = entry_mid + atr_dist * self.TP_ATR_MULTIPLIER
            else:  # SELL
                stop_loss = entry_mid + atr_dist * self.ATR_STOP_MULTIPLIER
                take_profit_1 = entry_mid - atr_dist * self.TP_ATR_MULTIPLIER
                if take_profit_1 <= 0.0:
                    return None

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
                regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                session=session,
                quality=quality,
                raw_features={
                    "ema_9": ema_9,
                    "ema_20": ema_20,
                    "htf_close": htf_close,
                    "htf_sma": htf_sma,
                    "last_close": last_close,
                    "atr_dist": atr_dist,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

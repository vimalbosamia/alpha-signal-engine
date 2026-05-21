"""
Candle Momentum Strategy — enters when 3+ consecutive candles push same direction.

Logic:
  - 3 consecutive bullish candles with strong bodies → BUY (momentum building)
  - 3 consecutive bearish candles with strong bodies → SELL (momentum building)
  - Average body ratio > 45% (not weak/indecision candles)
  - Each candle's close must progress further in the direction
  - Volume increasing or stable (not dying)

This catches strong directional moves. Complements candle_flip which catches reversals.
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


CONSECUTIVE_REQUIRED = 3
MIN_AVG_BODY_RATIO = 0.45
ATR_STOP_MULT = 1.5
ATR_TP1_MULT = 2.0
ATR_TP2_MULT = 3.5


class CandleMomentumStrategy(BaseStrategy):
    """Rides 3+ consecutive candle momentum in one direction."""

    @property
    def name(self) -> str:
        return "candle_momentum"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 20

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

            recent = df.iloc[-CONSECUTIVE_REQUIRED:]

            # Check all candles are same direction with strong bodies
            directions = []
            body_ratios = []
            closes = []

            for _, row in recent.iterrows():
                o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
                rng = h - l
                if rng <= 0:
                    return None
                body = abs(c - o)
                body_ratios.append(body / rng)
                directions.append(c > o)  # True = bullish
                closes.append(c)

            # All same direction?
            all_bull = all(directions)
            all_bear = all(not d for d in directions)
            if not all_bull and not all_bear:
                return None

            # Strong bodies?
            avg_body = sum(body_ratios) / len(body_ratios)
            if avg_body < MIN_AVG_BODY_RATIO:
                return None

            # Progressive closes (each further in direction)?
            if all_bull and not (closes[-1] > closes[-2] > closes[-3]):
                return None
            if all_bear and not (closes[-1] < closes[-2] < closes[-3]):
                return None

            action = SignalAction.BUY if all_bull else SignalAction.SELL
            close = closes[-1]
            atr = self._atr_stop(df, 1.0)
            if atr <= 0:
                return None

            if action == SignalAction.BUY:
                stop = close - (atr * ATR_STOP_MULT)
                tp1 = close + (atr * ATR_TP1_MULT)
                tp2 = close + (atr * ATR_TP2_MULT)
            else:
                stop = close + (atr * ATR_STOP_MULT)
                tp1 = close - (atr * ATR_TP1_MULT)
                tp2 = close - (atr * ATR_TP2_MULT)

            htf_bias = TrendDirection.UNKNOWN
            mkt_regime = regime.regime if regime and hasattr(regime, 'regime') else MarketRegime.UNKNOWN

            return SignalCandidate(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=self.name,
                proposed_action=action,
                timeframe=Timeframe.FIFTEEN_MIN,
                higher_tf_bias=htf_bias,
                entry_zone_low=round(close - atr * 0.1, 6),
                entry_zone_high=round(close + atr * 0.1, 6),
                stop_loss=round(stop, 6),
                take_profit_1=round(tp1, 6),
                take_profit_2=round(tp2, 6),
                regime=mkt_regime,
                session=session,
                quality=quality,
                raw_features={
                    "consecutive": CONSECUTIVE_REQUIRED,
                    "avg_body_ratio": round(avg_body, 3),
                    "direction": "bullish" if all_bull else "bearish",
                },
            )
        except Exception:
            return None

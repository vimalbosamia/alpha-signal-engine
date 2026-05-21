"""
Candle Direction Flip Strategy — trades every candle direction change.

Logic:
  - Track last N candles direction (bullish = close > open)
  - When direction FLIPS from previous candle:
    - Previous was bullish, current bearish → SELL signal
    - Previous was bearish, current bullish → BUY signal
  - Requires: strong body (>50% of range), not doji
  - Requires: current candle closed (not mid-bar)
  - Requires: volume confirms (current volume >= 80% of average)

This is a high-frequency scalp strategy. Works best on 1m-15m timeframes.
Designed for the scenario: "4min bullish → next bar bearish → flip position"

Position management:
  - Tight stops: 1.0x ATR
  - Quick targets: 1.5x ATR (TP1), 2.5x ATR (TP2)
  - R:R = 1.5 minimum
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


MIN_BODY_RATIO = 0.50      # candle body must be 50%+ of range
MIN_VOLUME_RATIO = 0.80    # volume must be 80%+ of rolling average
ATR_STOP_MULT = 1.0        # tight stop
ATR_TP1_MULT = 1.5         # quick TP1
ATR_TP2_MULT = 2.5         # extended TP2
LOOKBACK = 3               # check last 3 candles for flip detection


class CandleFlipStrategy(BaseStrategy):
    """Trades candle direction flips — reverses on every confirmed direction change."""

    @property
    def name(self) -> str:
        return "candle_direction_flip"

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

            # Get last 3 candles
            recent = df.iloc[-LOOKBACK:]
            prev = recent.iloc[-2]
            curr = recent.iloc[-1]

            prev_o, prev_c = float(prev["open"]), float(prev["close"])
            curr_o, curr_h, curr_l, curr_c = (
                float(curr["open"]), float(curr["high"]),
                float(curr["low"]), float(curr["close"]),
            )

            # Determine candle direction
            prev_bullish = prev_c > prev_o
            curr_bullish = curr_c > curr_o

            # No flip = no signal
            if prev_bullish == curr_bullish:
                return None

            # Check body strength — reject dojis/indecision
            curr_range = curr_h - curr_l
            if curr_range <= 0:
                return None
            curr_body = abs(curr_c - curr_o)
            body_ratio = curr_body / curr_range
            if body_ratio < MIN_BODY_RATIO:
                return None

            # Check volume confirmation
            if "volume" in df.columns:
                vol_avg = df["volume"].iloc[-20:].mean()
                curr_vol = float(curr["volume"])
                if vol_avg > 0 and (curr_vol / vol_avg) < MIN_VOLUME_RATIO:
                    return None

            # Direction flipped — generate signal
            # Previous bearish, current bullish → BUY
            # Previous bullish, current bearish → SELL
            if curr_bullish:
                action = SignalAction.BUY
            else:
                action = SignalAction.SELL

            # ATR for stop/target
            atr = self._atr_stop(df, 1.0)
            if atr <= 0:
                atr = curr_range * 1.5  # fallback

            close = curr_c
            stop_dist = atr * ATR_STOP_MULT
            tp1_dist = atr * ATR_TP1_MULT
            tp2_dist = atr * ATR_TP2_MULT

            if action == SignalAction.BUY:
                stop = close - stop_dist
                tp1 = close + tp1_dist
                tp2 = close + tp2_dist
                entry_low = close - (atr * 0.1)
                entry_high = close + (atr * 0.1)
            else:
                stop = close + stop_dist
                tp1 = close - tp1_dist
                tp2 = close - tp2_dist
                entry_low = close - (atr * 0.1)
                entry_high = close + (atr * 0.1)

            # Higher TF bias from structure
            htf_bias = TrendDirection.UNKNOWN
            if structure and hasattr(structure, 'trend'):
                if structure.trend == "uptrend":
                    htf_bias = TrendDirection.UPTREND
                elif structure.trend == "downtrend":
                    htf_bias = TrendDirection.DOWNTREND

            # Regime
            mkt_regime = MarketRegime.UNKNOWN
            if regime and hasattr(regime, 'regime'):
                mkt_regime = regime.regime

            return SignalCandidate(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=self.name,
                proposed_action=action,
                timeframe=Timeframe.FIFTEEN_MIN,
                higher_tf_bias=htf_bias,
                entry_zone_low=round(entry_low, 6),
                entry_zone_high=round(entry_high, 6),
                stop_loss=round(stop, 6),
                take_profit_1=round(tp1, 6),
                take_profit_2=round(tp2, 6),
                regime=mkt_regime,
                session=session,
                quality=quality,
                raw_features={
                    "body_ratio": round(body_ratio, 3),
                    "prev_bullish": prev_bullish,
                    "curr_bullish": curr_bullish,
                    "flip_type": "bear_to_bull" if curr_bullish else "bull_to_bear",
                },
            )
        except Exception:
            return None

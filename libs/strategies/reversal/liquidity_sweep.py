"""
Liquidity Sweep Reversal Strategy.

Logic:
  BUY  — price sweeps below the 20-bar low (wick below, close above) → liquidity
         grab → reversal long.
  SELL — price sweeps above the 20-bar high (wick above, close below) → bull trap
         → reversal short.

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


class LiquiditySweepReversalStrategy(BaseStrategy):
    """
    Detects liquidity sweeps (stop-hunt candles) and fades the false breakout.

    BUY setup  (sweep of 20-bar low):
      current low  < rolling_20_low  AND  current close > rolling_20_low

    SELL setup (sweep of 20-bar high):
      current high > rolling_20_high  AND  current close < rolling_20_high

    Stop loss:
      BUY:  below the sweep wick low (sweep_low - small buffer)
      SELL: above the sweep wick high (sweep_high + small buffer)

    Take profit: ATR * 3 from entry.
    """

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "liquidity_sweep"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

    # ── Thresholds ─────────────────────────────────────────────────────────────

    LOOKBACK: int = 20                  # bars for rolling high/low (excludes current)
    ATR_STOP_BUFFER: float = 0.1        # ATR fraction added below/above sweep extreme
    ATR_TP_MULTIPLIER: float = 3.0
    CONFIDENCE: float = 0.58

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

            # 3. Compute 20-bar rolling high and low from prior bars
            #    (exclude current bar so it can sweep those levels)
            prior = df.iloc[-(self.LOOKBACK + 1):-1]
            if len(prior) < self.LOOKBACK:
                return None

            rolling_high = float(prior["high"].max())
            rolling_low = float(prior["low"].min())

            # 4. Current bar values
            current = df.iloc[-1]
            current_high = float(current["high"])
            current_low = float(current["low"])
            current_close = float(current["close"])

            # 5. ATR for stop / TP sizing
            atr: float | None = indicators.atr if indicators is not None else None
            if atr is None or atr <= 0.0:
                atr = self._atr_stop(df, 1.0)  # fall back to rough ATR
            if atr <= 0.0:
                return None

            # 6. Detect sweep type
            buy_sweep = current_low < rolling_low and current_close > rolling_low
            sell_sweep = current_high > rolling_high and current_close < rolling_high

            if buy_sweep:
                return self._build_buy_candidate(
                    symbol, asset_class, df,
                    current_close, current_low, rolling_low,
                    atr, structure, regime, session, quality,
                )

            if sell_sweep:
                return self._build_sell_candidate(
                    symbol, asset_class, df,
                    current_close, current_high, rolling_high,
                    atr, structure, regime, session, quality,
                )

            return None

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private builders ───────────────────────────────────────────────────────

    def _build_buy_candidate(
        self,
        symbol: str,
        asset_class: AssetClass,
        df: pd.DataFrame,
        close: float,
        sweep_low: float,
        rolling_low: float,
        atr: float,
        structure: MarketStructure | None,
        regime: RegimeAnalysis | None,
        session: SessionState | None,
        quality: DataQualityReport | None,
    ) -> SignalCandidate | None:
        """Build a BUY signal after a low sweep + reclaim."""
        # Stop just below the sweep wick
        stop_loss = sweep_low - self.ATR_STOP_BUFFER * atr
        if stop_loss <= 0.0:
            return None

        # Entry around the current close
        entry_low = close * 0.9998
        entry_high = close * 1.0002
        entry_mid = (entry_low + entry_high) / 2.0

        if stop_loss >= entry_mid:
            return None

        take_profit_1 = entry_mid + self.ATR_TP_MULTIPLIER * atr

        higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN

        return SignalCandidate(
            symbol=symbol,
            asset_class=asset_class,
            strategy_name=self.name,
            proposed_action=SignalAction.BUY,
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
                "sweep_low": sweep_low,
                "rolling_low": rolling_low,
                "close": close,
                "atr": atr,
                "confidence": self.CONFIDENCE,
            },
        )

    def _build_sell_candidate(
        self,
        symbol: str,
        asset_class: AssetClass,
        df: pd.DataFrame,
        close: float,
        sweep_high: float,
        rolling_high: float,
        atr: float,
        structure: MarketStructure | None,
        regime: RegimeAnalysis | None,
        session: SessionState | None,
        quality: DataQualityReport | None,
    ) -> SignalCandidate | None:
        """Build a SELL signal after a high sweep + rejection."""
        # Stop just above the sweep wick
        stop_loss = sweep_high + self.ATR_STOP_BUFFER * atr
        if stop_loss <= 0.0:
            return None

        # Entry around the current close
        entry_low = close * 0.9998
        entry_high = close * 1.0002
        entry_mid = (entry_low + entry_high) / 2.0

        if stop_loss <= entry_mid:
            return None

        take_profit_1 = entry_mid - self.ATR_TP_MULTIPLIER * atr
        if take_profit_1 <= 0.0:
            return None

        higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN

        return SignalCandidate(
            symbol=symbol,
            asset_class=asset_class,
            strategy_name=self.name,
            proposed_action=SignalAction.SELL,
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
                "sweep_high": sweep_high,
                "rolling_high": rolling_high,
                "close": close,
                "atr": atr,
                "confidence": self.CONFIDENCE,
            },
        )

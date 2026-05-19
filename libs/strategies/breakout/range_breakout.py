"""
Range Breakout Strategy.

Logic: Detect a compressed price range (20-bar high - 20-bar low < ATR*4).
Buy when close breaks above range high; sell when close breaks below range low.
Volume confirmation required (> 1.5x average on breakout bar).

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


class RangeBreakoutStrategy(BaseStrategy):
    """Breakout from a compressed range with volume confirmation."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "range_breakout"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 30

    # ── Thresholds ─────────────────────────────────────────────────────────────

    RANGE_LOOKBACK: int = 20             # bars used to define the range
    RANGE_ATR_MULTIPLIER: float = 4.0   # range < ATR * this → compressed
    MIN_REL_VOL: float = 1.5            # breakout bar must exceed this relative volume
    BASE_CONFIDENCE: float = 0.60

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

            # 3. Compute range over the last RANGE_LOOKBACK bars (excluding current)
            lookback_df = df.iloc[-(self.RANGE_LOOKBACK + 1):-1]
            range_high = float(lookback_df["high"].max())
            range_low = float(lookback_df["low"].min())
            range_width = range_high - range_low

            if range_width <= 0.0:
                return None

            # 4. Derive ATR for compression check
            atr = self._get_atr(df, indicators)
            if atr is None or atr <= 0.0:
                return None

            # 5. Require compressed range
            if range_width >= atr * self.RANGE_ATR_MULTIPLIER:
                return None

            # 6. Volume confirmation on the breakout bar
            last_bar = df.iloc[-1]
            last_close = float(last_bar["close"])
            last_rel_vol = float(last_bar["relative_volume"]) if "relative_volume" in df.columns else None

            if last_rel_vol is not None and last_rel_vol < self.MIN_REL_VOL:
                return None

            # 7. Determine breakout direction
            broke_above = last_close > range_high
            broke_below = last_close < range_low

            if not (broke_above or broke_below):
                return None

            action = SignalAction.BUY if broke_above else SignalAction.SELL
            range_mid = (range_high + range_low) / 2.0

            # 8. Entry zone, stop, and take-profit
            if action == SignalAction.BUY:
                entry_low = last_close
                entry_high = last_close * 1.001
                stop_loss = range_mid
                if stop_loss >= last_close:
                    return None
                stop_distance = last_close - stop_loss
                take_profit_1 = last_close + range_width
            else:
                entry_low = last_close * 0.999
                entry_high = last_close
                stop_loss = range_mid
                if stop_loss <= last_close:
                    return None
                stop_distance = stop_loss - last_close
                take_profit_1 = last_close - range_width

            if stop_distance <= 0.0:
                return None

            # 9. HTF bias
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
                    "range_high": range_high,
                    "range_low": range_low,
                    "range_width": range_width,
                    "atr": atr,
                    "last_close": last_close,
                    "relative_volume": last_rel_vol,
                    "confidence": self.BASE_CONFIDENCE,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_atr(
        self,
        df: pd.DataFrame,
        indicators: IndicatorSnapshot | None,
    ) -> float | None:
        """Return ATR from indicators snapshot if available, else compute from df."""
        if indicators is not None and indicators.atr is not None:
            return indicators.atr
        # Fallback: rolling 14-bar average true range
        if len(df) < 14:
            return None
        tr = (df["high"] - df["low"]).abs().rolling(14).mean().iloc[-1]
        import math
        if tr is not None and not math.isnan(float(tr)):
            return float(tr)
        return None

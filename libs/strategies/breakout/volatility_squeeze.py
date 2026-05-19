"""
Volatility Contraction Breakout (Volatility Squeeze) Strategy.

Logic: Bollinger Band width contracts to a 20-bar low (squeeze). When the BB
width then expands (current > previous), a breakout is signalled.  Candle
direction determines trade side: bullish bar → BUY, bearish bar → SELL.

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


class VolatilitySqueezeStrategy(BaseStrategy):
    """
    Bollinger Band squeeze expansion breakout.

    Squeeze: bb_width is in the bottom 20% of its 20-bar rolling range.
    Expansion: current bb_width > previous bb_width * 1.1 (expansion trigger).
    Direction: determined by the current candle (bullish → BUY, bearish → SELL).
    """

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "volatility_squeeze"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

    # ── Thresholds ─────────────────────────────────────────────────────────────

    BB_WIDTH_LOOKBACK: int = 20         # bars to compute the squeeze range
    SQUEEZE_PERCENTILE: float = 0.20    # bb_width must be in bottom 20%
    EXPANSION_FACTOR: float = 1.1       # current bb_width > prev * this factor
    ATR_STOP_MULTIPLIER: float = 2.0
    ATR_TP_MULTIPLIER: float = 4.0
    CONFIDENCE: float = 0.60

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

            # 3. Require Bollinger Band width data from indicators
            if indicators is None:
                return None

            bb_width = indicators.bb_width
            if bb_width is None:
                return None

            # 4. Compute 20-bar bb_width history from df if available, else use
            #    the current bb_width approximation from indicators only.
            squeeze_confirmed = self._is_in_squeeze(df, bb_width)
            if not squeeze_confirmed:
                return None

            # 5. Check expansion: current bb_width > prev bb_width * factor
            if not self._is_expanding(df, bb_width):
                return None

            # 6. Determine direction from current candle
            last_bar = df.iloc[-1]
            last_close = float(last_bar["close"])
            last_open = float(last_bar["open"])
            is_bullish_bar = last_close > last_open

            if is_bullish_bar:
                proposed_action = SignalAction.BUY
            else:
                proposed_action = SignalAction.SELL

            # 7. Compute ATR-based stops
            atr = indicators.atr
            if atr is None or atr <= 0.0:
                return None

            if proposed_action == SignalAction.BUY:
                stop_loss = last_close - self.ATR_STOP_MULTIPLIER * atr
                take_profit_1 = last_close + self.ATR_TP_MULTIPLIER * atr
            else:
                stop_loss = last_close + self.ATR_STOP_MULTIPLIER * atr
                take_profit_1 = last_close - self.ATR_TP_MULTIPLIER * atr

            if stop_loss <= 0.0:
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
                    "bb_width": bb_width,
                    "atr": atr,
                    "is_bullish_bar": is_bullish_bar,
                    "confidence": self.CONFIDENCE,
                },
            )

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _is_in_squeeze(self, df: pd.DataFrame, current_bb_width: float) -> bool:
        """
        Return True when the current bb_width is in the bottom SQUEEZE_PERCENTILE
        of the last BB_WIDTH_LOOKBACK bars.

        If the 'bb_width' column is absent from df we fall back to checking
        whether the indicators bb_width value is below a rough proxy threshold
        derived from ATR / close (always approximate).  The column-based path
        is preferred and much more accurate.
        """
        if "bb_width" in df.columns:
            window = df["bb_width"].iloc[-self.BB_WIDTH_LOOKBACK:].dropna()
            if len(window) < 5:  # not enough history
                return False
            threshold = window.quantile(self.SQUEEZE_PERCENTILE)
            return float(current_bb_width) <= float(threshold) * 1.2  # slight buffer

        # Fallback: if no column, assume squeeze when bb_width is very narrow
        # (below 2% relative width — heuristic only)
        return current_bb_width < 0.02

    def _is_expanding(self, df: pd.DataFrame, current_bb_width: float) -> bool:
        """
        Return True when the current bb_width exceeds the previous bar's
        bb_width by at least EXPANSION_FACTOR, signalling the squeeze is
        releasing.

        Uses the 'bb_width' column if present; otherwise cannot confirm
        expansion and returns False (conservative).
        """
        if "bb_width" not in df.columns:
            return False

        series = df["bb_width"].dropna()
        if len(series) < 2:
            return False

        prev_bb_width = float(series.iloc[-2])
        if prev_bb_width <= 0.0:
            return False

        return float(current_bb_width) > prev_bb_width * self.EXPANSION_FACTOR

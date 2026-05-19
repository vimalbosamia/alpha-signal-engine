"""
Break and Retest Strategy.

Logic: After a breakout above resistance (5-bar high broken 3-10 bars ago),
price pulls back and retests the broken level.
BUY when price retests (touches within ATR*0.5 of the level) and current bar
is bullish (close > open).

For SELL: breakout below 5-bar low, then retest from below.

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


class BreakRetestStrategy(BaseStrategy):
    """Buy/sell the retest of a freshly broken level."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "break_and_retest"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 25

    # ── Thresholds ─────────────────────────────────────────────────────────────

    RESISTANCE_LOOKBACK: int = 5         # bars used to define the pre-break resistance
    BREAKOUT_MIN_BARS_AGO: int = 3       # breakout happened at least this many bars ago
    BREAKOUT_MAX_BARS_AGO: int = 10      # but no more than this many bars ago
    RETEST_ATR_TOLERANCE: float = 0.5   # retest must be within ATR * this of the level
    ATR_STOP_MULTIPLIER: float = 1.0    # stop below/above retest level ± ATR
    TP_ATR_MULTIPLIER: float = 3.0      # take-profit = entry ± ATR * this
    BASE_CONFIDENCE: float = 0.63

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

            atr = self._get_atr(df, indicators)
            if atr is None or atr <= 0.0:
                return None

            last_bar = df.iloc[-1]
            last_close = float(last_bar["close"])
            last_open = float(last_bar["open"])
            current_bar_is_bullish = last_close > last_open
            current_bar_is_bearish = last_close < last_open

            # 3a. Check for BUY setup: prior breakout above resistance, now retesting
            buy_candidate = self._check_buy_setup(
                df, last_close, atr,
                current_bar_is_bullish,
            )
            if buy_candidate is not None:
                level_price, retest_low, retest_high, stop_loss, take_profit_1 = buy_candidate
                higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN
                return SignalCandidate(
                    symbol=symbol,
                    asset_class=asset_class,
                    strategy_name=self.name,
                    proposed_action=SignalAction.BUY,
                    timeframe=Timeframe.FIFTEEN_MIN,
                    higher_tf_bias=higher_tf_bias,
                    entry_zone_low=retest_low,
                    entry_zone_high=retest_high,
                    stop_loss=stop_loss,
                    take_profit_1=take_profit_1,
                    regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "level_price": level_price,
                        "last_close": last_close,
                        "atr": atr,
                        "confidence": self.BASE_CONFIDENCE,
                        "setup": "break_and_retest_bull",
                    },
                )

            # 3b. Check for SELL setup: prior breakdown below support, now retesting
            if current_bar_is_bearish:
                sell_candidate = self._check_sell_setup(df, last_close, atr)
                if sell_candidate is not None:
                    level_price, retest_low, retest_high, stop_loss, take_profit_1 = sell_candidate
                    higher_tf_bias = structure.trend if structure is not None else TrendDirection.UNKNOWN
                    return SignalCandidate(
                        symbol=symbol,
                        asset_class=asset_class,
                        strategy_name=self.name,
                        proposed_action=SignalAction.SELL,
                        timeframe=Timeframe.FIFTEEN_MIN,
                        higher_tf_bias=higher_tf_bias,
                        entry_zone_low=retest_low,
                        entry_zone_high=retest_high,
                        stop_loss=stop_loss,
                        take_profit_1=take_profit_1,
                        regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                        session=session,
                        quality=quality,
                        raw_features={
                            "level_price": level_price,
                            "last_close": last_close,
                            "atr": atr,
                            "confidence": self.BASE_CONFIDENCE,
                            "setup": "break_and_retest_bear",
                        },
                    )

            return None

        except Exception:  # noqa: BLE001 — never propagate
            return None

    # ── Private helpers ────────────────────────────────────────────────────────

    def _check_buy_setup(
        self,
        df: pd.DataFrame,
        last_close: float,
        atr: float,
        current_bar_is_bullish: bool,
    ) -> tuple[float, float, float, float, float] | None:
        """
        Scan for a bull break-and-retest setup.

        A breakout bar broke above the 5-bar high 3-10 bars ago.
        Current bar is close to that level (within ATR*0.5) and is bullish.

        Returns (level_price, entry_low, entry_high, stop_loss, take_profit_1)
        or None if no setup found.
        """
        if not current_bar_is_bullish:
            return None

        tolerance = atr * self.RETEST_ATR_TOLERANCE
        n = len(df)

        for bars_ago in range(self.BREAKOUT_MIN_BARS_AGO, self.BREAKOUT_MAX_BARS_AGO + 1):
            breakout_idx = n - 1 - bars_ago
            if breakout_idx < self.RESISTANCE_LOOKBACK:
                break

            # Resistance level = 5-bar high just before the breakout bar
            pre_break_df = df.iloc[breakout_idx - self.RESISTANCE_LOOKBACK:breakout_idx]
            resistance = float(pre_break_df["high"].max())

            # Confirm the breakout bar closed above that resistance
            breakout_bar_close = float(df.iloc[breakout_idx]["close"])
            if breakout_bar_close <= resistance:
                continue

            # Confirm price has pulled back near the level
            if abs(last_close - resistance) > tolerance:
                continue

            # Build trade parameters
            entry_low = last_close * 0.9995
            entry_high = last_close * 1.0005
            stop_loss = resistance - atr * self.ATR_STOP_MULTIPLIER
            if stop_loss <= 0.0 or stop_loss >= last_close:
                continue
            take_profit_1 = last_close + atr * self.TP_ATR_MULTIPLIER

            return (resistance, entry_low, entry_high, stop_loss, take_profit_1)

        return None

    def _check_sell_setup(
        self,
        df: pd.DataFrame,
        last_close: float,
        atr: float,
    ) -> tuple[float, float, float, float, float] | None:
        """
        Scan for a bear break-and-retest setup.

        A breakdown bar broke below the 5-bar low 3-10 bars ago.
        Current bar is close to that level (within ATR*0.5) and is bearish.

        Returns (level_price, entry_low, entry_high, stop_loss, take_profit_1)
        or None if no setup found.
        """
        tolerance = atr * self.RETEST_ATR_TOLERANCE
        n = len(df)

        for bars_ago in range(self.BREAKOUT_MIN_BARS_AGO, self.BREAKOUT_MAX_BARS_AGO + 1):
            breakout_idx = n - 1 - bars_ago
            if breakout_idx < self.RESISTANCE_LOOKBACK:
                break

            # Support level = 5-bar low just before the breakdown bar
            pre_break_df = df.iloc[breakout_idx - self.RESISTANCE_LOOKBACK:breakout_idx]
            support = float(pre_break_df["low"].min())

            # Confirm the breakdown bar closed below that support
            breakdown_bar_close = float(df.iloc[breakout_idx]["close"])
            if breakdown_bar_close >= support:
                continue

            # Confirm price has rallied back near the level
            if abs(last_close - support) > tolerance:
                continue

            # Build trade parameters
            entry_low = last_close * 0.9995
            entry_high = last_close * 1.0005
            stop_loss = support + atr * self.ATR_STOP_MULTIPLIER
            if stop_loss <= last_close:
                continue
            take_profit_1 = last_close - atr * self.TP_ATR_MULTIPLIER
            if take_profit_1 <= 0.0:
                continue

            return (support, entry_low, entry_high, stop_loss, take_profit_1)

        return None

    def _get_atr(
        self,
        df: pd.DataFrame,
        indicators: IndicatorSnapshot | None,
    ) -> float | None:
        """Return ATR from snapshot if available, else compute from df."""
        if indicators is not None and indicators.atr is not None:
            return indicators.atr
        if len(df) < 14:
            return None
        tr = (df["high"] - df["low"]).abs().rolling(14).mean().iloc[-1]
        import math
        if tr is not None and not math.isnan(float(tr)):
            return float(tr)
        return None

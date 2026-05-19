"""
Opening Range Breakout Strategy.

Logic: Uses first 3 bars as the "opening range" (highest high, lowest low of
bars 0-2).  BUY when a later bar closes above the range high.  SELL when a
later bar closes below the range low.  Only fires once per session — as soon
as a prior bar after index 2 already broke the range, the setup is stale.

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


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """Breakout above/below the first-N-bar opening range."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "opening_range_breakout"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 10

    # ── Thresholds ─────────────────────────────────────────────────────────────

    OPENING_RANGE_BARS: int = 3          # bars that define the opening range
    CONFIDENCE: float = 0.57
    ATR_STOP_MULTIPLIER: float = 2.0

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

            # 3. Build opening range from first OPENING_RANGE_BARS bars
            or_bars = df.iloc[: self.OPENING_RANGE_BARS]
            or_high = float(or_bars["high"].max())
            or_low = float(or_bars["low"].min())
            or_width = or_high - or_low

            if or_width <= 0.0:
                return None

            # 4. Post-range bars (index > 2)
            post_range = df.iloc[self.OPENING_RANGE_BARS :]
            if post_range.empty:
                return None

            # 5. Ensure signal fires only once per session:
            #    if any bar before the last already broke, skip.
            prior_bars = post_range.iloc[:-1]
            already_broke_high = (prior_bars["close"] > or_high).any()
            already_broke_low = (prior_bars["close"] < or_low).any()
            if already_broke_high or already_broke_low:
                return None

            last = post_range.iloc[-1]
            last_close = float(last["close"])

            # 6. BUY: close breaks above opening range high
            if last_close > or_high:
                stop_loss = or_low
                if stop_loss <= 0.0:
                    return None
                entry_low = last_close
                entry_high = last_close * 1.001
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss >= entry_mid:
                    return None
                take_profit_1 = or_high + or_width  # project range width upward

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
                    regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "or_high": or_high,
                        "or_low": or_low,
                        "or_width": or_width,
                        "last_close": last_close,
                        "confidence": self.CONFIDENCE,
                    },
                )

            # 7. SELL: close breaks below opening range low
            if last_close < or_low:
                stop_loss = or_high
                entry_low = last_close * 0.999
                entry_high = last_close
                entry_mid = (entry_low + entry_high) / 2.0
                if stop_loss <= entry_mid:
                    return None
                take_profit_1 = or_low - or_width  # project range width downward
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
                    regime=regime.regime if regime is not None else MarketRegime.UNKNOWN,
                    session=session,
                    quality=quality,
                    raw_features={
                        "or_high": or_high,
                        "or_low": or_low,
                        "or_width": or_width,
                        "last_close": last_close,
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

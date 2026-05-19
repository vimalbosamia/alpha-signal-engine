"""
Trend Following Strategy.

Logic (all conditions required — strong trend confirmation):
  BUY : EMA9 > EMA20 > EMA50  AND  close > EMA9  AND  ADX > 25  AND  RSI > 50
  SELL: EMA9 < EMA20 < EMA50  AND  close < EMA9  AND  ADX > 25  AND  RSI < 50

Stop: below/above EMA50.
TP  : ATR * 4 from entry.
Confidence: 0.68

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


class TrendFollowingStrategy(BaseStrategy):
    """Triple EMA stack + ADX + RSI trend-following strategy."""

    # ── Identity ───────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "trend_following"

    @property
    def supported_asset_classes(self) -> list[AssetClass]:
        return [AssetClass.STOCK, AssetClass.CRYPTO]

    @property
    def min_bars_required(self) -> int:
        return 55

    # ── Thresholds ─────────────────────────────────────────────────────────────

    MIN_ADX: float = 25.0
    RSI_BULL_THRESHOLD: float = 50.0
    RSI_BEAR_THRESHOLD: float = 50.0
    ATR_TP_MULTIPLIER: float = 4.0
    CONFIDENCE: float = 0.68

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

            # 3. Resolve indicator values
            ema9 = self._resolve_ema9(df, indicators)
            ema20 = self._resolve_ema20(df, indicators)
            ema50 = self._resolve_ema50(df, indicators)
            adx = indicators.adx if (indicators and indicators.adx is not None) else None
            rsi = indicators.rsi if (indicators and indicators.rsi is not None) else None

            # All indicators must be present
            if any(v is None for v in (ema9, ema20, ema50, adx, rsi)):
                return None

            # 4. ADX gate — trending market required
            if adx < self.MIN_ADX:  # type: ignore[operator]
                return None

            last_close = float(df["close"].iloc[-1])
            atr = self._get_atr(df, indicators)
            if atr is None or atr <= 0.0:
                return None

            market_regime = regime.regime if regime is not None else MarketRegime.UNKNOWN
            tp_dist = atr * self.ATR_TP_MULTIPLIER

            # ── BUY: full bullish EMA stack ───────────────────────────────────
            if (
                ema9 > ema20  # type: ignore[operator]
                and ema20 > ema50  # type: ignore[operator]
                and last_close > ema9  # type: ignore[operator]
                and rsi > self.RSI_BULL_THRESHOLD  # type: ignore[operator]
            ):
                stop_loss = float(ema50)  # type: ignore[arg-type]
                if stop_loss <= 0.0 or stop_loss >= last_close:
                    return None

                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                take_profit_1 = entry_mid + tp_dist

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
                    regime=market_regime,
                    session=session,
                    quality=quality,
                    raw_features={
                        "last_close": last_close,
                        "ema9": ema9,
                        "ema20": ema20,
                        "ema50": ema50,
                        "adx": adx,
                        "rsi": rsi,
                        "atr": atr,
                        "confidence": self.CONFIDENCE,
                    },
                )

            # ── SELL: full bearish EMA stack ──────────────────────────────────
            if (
                ema9 < ema20  # type: ignore[operator]
                and ema20 < ema50  # type: ignore[operator]
                and last_close < ema9  # type: ignore[operator]
                and rsi < self.RSI_BEAR_THRESHOLD  # type: ignore[operator]
            ):
                stop_loss = float(ema50)  # type: ignore[arg-type]
                if stop_loss <= last_close:
                    return None

                entry_low = last_close * 0.9995
                entry_high = last_close * 1.0005
                entry_mid = (entry_low + entry_high) / 2.0

                take_profit_1 = entry_mid - tp_dist
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
                    regime=market_regime,
                    session=session,
                    quality=quality,
                    raw_features={
                        "last_close": last_close,
                        "ema9": ema9,
                        "ema20": ema20,
                        "ema50": ema50,
                        "adx": adx,
                        "rsi": rsi,
                        "atr": atr,
                        "confidence": self.CONFIDENCE,
                    },
                )

        except Exception:  # noqa: BLE001 — never propagate
            return None

        return None

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_ema9(df: pd.DataFrame, indicators: IndicatorSnapshot | None) -> float | None:
        if indicators is not None and indicators.ema_9 is not None:
            return indicators.ema_9
        series = df["close"].ewm(span=9, adjust=False).mean()
        val = series.iloc[-1]
        return float(val) if not pd.isna(val) else None

    @staticmethod
    def _resolve_ema20(df: pd.DataFrame, indicators: IndicatorSnapshot | None) -> float | None:
        if indicators is not None and indicators.ema_20 is not None:
            return indicators.ema_20
        series = df["close"].ewm(span=20, adjust=False).mean()
        val = series.iloc[-1]
        return float(val) if not pd.isna(val) else None

    @staticmethod
    def _resolve_ema50(df: pd.DataFrame, indicators: IndicatorSnapshot | None) -> float | None:
        if indicators is not None and indicators.ema_50 is not None:
            return indicators.ema_50
        series = df["close"].ewm(span=50, adjust=False).mean()
        val = series.iloc[-1]
        return float(val) if not pd.isna(val) else None

    @staticmethod
    def _get_atr(df: pd.DataFrame, indicators: IndicatorSnapshot | None) -> float | None:
        if indicators is not None and indicators.atr is not None:
            return indicators.atr
        tr = (df["high"] - df["low"]).abs().rolling(14).mean().iloc[-1]
        if pd.isna(tr) or tr <= 0.0:
            return None
        return float(tr)

    @staticmethod
    def _htf_bias(structure: MarketStructure | None) -> TrendDirection:
        return structure.trend if structure is not None else TrendDirection.UNKNOWN

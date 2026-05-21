"""
MarketContextCollector — captures full market context at trade entry time.

Rich context from signals, indicators, volume, and macro state becomes
training data for all self-training subsystems.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional


@dataclass
class MarketContext:
    """
    Full market context snapshot captured at trade entry.

    All fields are Optional with None defaults so partial contexts
    can be built incrementally and merged.
    """

    # ── Signal fields ──────────────────────────────────────────────────────────
    timeframe: Optional[str] = None
    market_regime: Optional[str] = None
    entry_confidence: Optional[float] = None
    entry_confluence_score: Optional[float] = None
    entry_bias_score: Optional[float] = None
    setup_grade: Optional[str] = None
    entry_patterns: Optional[list] = field(default=None)

    # ── Indicator fields ───────────────────────────────────────────────────────
    entry_rsi: Optional[float] = None
    entry_macd_histogram: Optional[float] = None
    entry_ema_structure: Optional[str] = None
    entry_atr: Optional[float] = None
    entry_atr_pct: Optional[float] = None
    entry_bollinger_pct_b: Optional[float] = None
    entry_adx: Optional[float] = None
    entry_vwap_deviation_pct: Optional[float] = None

    # ── Volume fields ──────────────────────────────────────────────────────────
    entry_volume_relative: Optional[float] = None
    entry_volume_trend: Optional[str] = None
    entry_orderflow_bias: Optional[str] = None
    entry_liquidity_state: Optional[str] = None

    # ── Macro fields ───────────────────────────────────────────────────────────
    entry_news_sentiment: Optional[str] = None
    entry_macro_environment: Optional[str] = None
    entry_dxy_trend: Optional[str] = None
    entry_bond_yield_trend: Optional[str] = None
    entry_btc_dominance_trend: Optional[str] = None
    entry_fear_greed_index: Optional[int] = None
    entry_funding_rate: Optional[float] = None
    entry_open_interest_trend: Optional[str] = None

    # ── Session / execution fields ─────────────────────────────────────────────
    entry_session: Optional[str] = None
    entry_spread_pct: Optional[float] = None
    leverage: Optional[float] = None
    trading_mode: Optional[str] = None
    slippage_pct: Optional[float] = None
    execution_latency_ms: Optional[int] = None

    def to_dict(self) -> dict:
        """Return only non-None fields as a plain dict."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }


def _enum_value(obj: Any) -> Any:
    """Safely extract .value from an enum-like object, else return as-is."""
    return obj.value if hasattr(obj, "value") else obj


class MarketContextCollector:
    """
    Static factory methods that build partial MarketContext objects
    from different data sources.  Use merge() to combine partials.
    """

    @staticmethod
    def from_signal(signal: Any) -> MarketContext:
        """
        Extract signal-level context from a SignalOutput (or similar) object.

        Enum fields are coerced to their .value string representations.
        """
        confluence_score: Optional[float] = None
        raw_confluence = getattr(signal, "confluence", None)
        if raw_confluence is not None:
            confluence_score = getattr(raw_confluence, "weighted_total", None)

        bias_score: Optional[float] = None
        bias_scores_obj = getattr(signal, "bias_scores", None)
        if bias_scores_obj is not None:
            bullish = getattr(bias_scores_obj, "bullish_score", None)
            bearish = getattr(bias_scores_obj, "bearish_score", None)
            if bullish is not None and bearish is not None:
                bias_score = float(bullish) - float(bearish)

        raw_patterns = getattr(signal, "patterns_detected", None)
        entry_patterns: Optional[list] = list(raw_patterns) if raw_patterns is not None else None

        return MarketContext(
            timeframe=_enum_value(getattr(signal, "timeframe", None)),
            market_regime=_enum_value(getattr(signal, "market_regime", None)),
            entry_confidence=getattr(signal, "confidence", None),
            entry_confluence_score=confluence_score,
            entry_bias_score=bias_score,
            setup_grade=_enum_value(getattr(signal, "setup_grade", None)),
            entry_patterns=entry_patterns,
        )

    @staticmethod
    def from_indicators(
        rsi: Optional[float] = None,
        macd_histogram: Optional[float] = None,
        ema_structure: Optional[str] = None,
        atr: Optional[float] = None,
        atr_pct: Optional[float] = None,
        bollinger_pct_b: Optional[float] = None,
        adx: Optional[float] = None,
        vwap_deviation_pct: Optional[float] = None,
    ) -> MarketContext:
        """Build a partial MarketContext populated with indicator values."""
        return MarketContext(
            entry_rsi=rsi,
            entry_macd_histogram=macd_histogram,
            entry_ema_structure=ema_structure,
            entry_atr=atr,
            entry_atr_pct=atr_pct,
            entry_bollinger_pct_b=bollinger_pct_b,
            entry_adx=adx,
            entry_vwap_deviation_pct=vwap_deviation_pct,
        )

    @staticmethod
    def from_volume(
        relative: Optional[float] = None,
        trend: Optional[str] = None,
        orderflow_bias: Optional[str] = None,
        liquidity_state: Optional[str] = None,
    ) -> MarketContext:
        """Build a partial MarketContext populated with volume data."""
        return MarketContext(
            entry_volume_relative=relative,
            entry_volume_trend=trend,
            entry_orderflow_bias=orderflow_bias,
            entry_liquidity_state=liquidity_state,
        )

    @staticmethod
    def from_macro(
        news_sentiment: Optional[str] = None,
        macro_environment: Optional[str] = None,
        dxy_trend: Optional[str] = None,
        bond_yield_trend: Optional[str] = None,
        btc_dominance_trend: Optional[str] = None,
        fear_greed_index: Optional[int] = None,
        funding_rate: Optional[float] = None,
        open_interest_trend: Optional[str] = None,
    ) -> MarketContext:
        """Build a partial MarketContext populated with macro data."""
        return MarketContext(
            entry_news_sentiment=news_sentiment,
            entry_macro_environment=macro_environment,
            entry_dxy_trend=dxy_trend,
            entry_bond_yield_trend=bond_yield_trend,
            entry_btc_dominance_trend=btc_dominance_trend,
            entry_fear_greed_index=fear_greed_index,
            entry_funding_rate=funding_rate,
            entry_open_interest_trend=open_interest_trend,
        )

    @staticmethod
    def merge(*contexts: MarketContext) -> MarketContext:
        """
        Merge multiple partial MarketContext objects into one.

        Later contexts override earlier ones for any field that is not None.
        """
        merged = MarketContext()
        for ctx in contexts:
            for f in fields(ctx):
                value = getattr(ctx, f.name)
                if value is not None:
                    object.__setattr__(merged, f.name, value)
        return merged

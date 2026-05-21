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

    # Map IndicatorResult.name → MarketContext field name
    _INDICATOR_MAP: dict[str, str] = {
        "rsi": "entry_rsi",
        "RSI": "entry_rsi",
        "macd_histogram": "entry_macd_histogram",
        "MACD Histogram": "entry_macd_histogram",
        "atr": "entry_atr",
        "ATR": "entry_atr",
        "adx": "entry_adx",
        "ADX": "entry_adx",
        "bb_pct_b": "entry_bollinger_pct_b",
        "Bollinger %B": "entry_bollinger_pct_b",
        "vwap_deviation": "entry_vwap_deviation_pct",
        "VWAP Deviation": "entry_vwap_deviation_pct",
    }

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
    def from_signal_indicators(signal: Any) -> MarketContext:
        """Extract indicator values from signal.indicator_results list.

        Each IndicatorResult has .name and .value — maps known names to
        MarketContext fields via _INDICATOR_MAP.
        """
        indicator_results = getattr(signal, "indicator_results", None)
        if not indicator_results:
            return MarketContext()

        kwargs: dict[str, Any] = {}
        ema_structure: Optional[str] = None

        for ir in indicator_results:
            name = getattr(ir, "name", "")
            value = getattr(ir, "value", None)
            if value is None:
                continue

            mapped = MarketContextCollector._INDICATOR_MAP.get(name)
            if mapped:
                kwargs[mapped] = float(value)

            # Derive EMA structure from bias field
            if name in ("ema_structure", "EMA Structure"):
                ema_structure = _enum_value(getattr(ir, "bias", None)) or str(value)

        if ema_structure:
            kwargs["entry_ema_structure"] = ema_structure

        # Derive ATR% if we have ATR and can get entry price
        atr_val = kwargs.get("entry_atr")
        if atr_val and atr_val > 0:
            entry_mid = None
            entry_low = getattr(signal, "entry_zone_low", None)
            entry_high = getattr(signal, "entry_zone_high", None)
            if entry_low and entry_high:
                entry_mid = (entry_low + entry_high) / 2.0
            if entry_mid and entry_mid > 0:
                kwargs["entry_atr_pct"] = round(atr_val / entry_mid * 100, 4)

        return MarketContext(**kwargs)

    @staticmethod
    def from_signal_full(signal: Any) -> MarketContext:
        """Build complete context by merging signal + indicator_results.

        Single call that extracts everything available on a SignalOutput.
        """
        ctx_signal = MarketContextCollector.from_signal(signal)
        ctx_indicators = MarketContextCollector.from_signal_indicators(signal)

        # Session detection from signal's session_status
        session_str = _enum_value(getattr(signal, "session_status", None))

        # Volume/macro from risk_result if available
        risk = getattr(signal, "risk_result", None)
        spread_pct = None
        if risk:
            spread_warn = getattr(risk, "spread_warning", "")
            if spread_warn:
                # Try extracting numeric spread from warning text
                try:
                    import re
                    m = re.search(r"(\d+\.?\d*)", spread_warn)
                    if m:
                        spread_pct = float(m.group(1))
                except Exception:
                    pass

        ctx_session = MarketContext(
            entry_session=session_str,
            entry_spread_pct=spread_pct,
        )

        return MarketContextCollector.merge(ctx_signal, ctx_indicators, ctx_session)

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

"""Pipeline stage functions — extracted from pipeline.py for testability."""
from __future__ import annotations


def compute_symbol_bias(df, indicators, htf_structure, regime, pattern_results):
    """
    Compute directional bias for a symbol.

    Returns (BiasResult, allowed_action, htf_bias_str, indicator_bias, deep_structure).
    """
    from libs.analysis.indicators.bias import IndicatorBiasAnalyzer
    from libs.analysis.structure.market_structure import MarketStructureAnalyzer
    from libs.analysis.bias.engine import BullBearBiasEngine, BiasInput
    from libs.core.models.domain import SignalAction

    # Indicator bias
    last_close = float(df["close"].iloc[-1])
    rel_vol = float(df["relative_volume"].iloc[-1]) if "relative_volume" in df else 1.0
    is_bull = bool(df["is_bullish"].iloc[-1]) if "is_bullish" in df else True

    indicator_bias = IndicatorBiasAnalyzer().analyze_all(
        rsi=indicators.rsi or 50,
        rsi_prev=indicators.rsi_prev or 50,
        macd_line=indicators.macd_line or 0,
        macd_signal=indicators.macd_signal or 0,
        histogram=indicators.macd_histogram or 0,
        histogram_prev=indicators.macd_histogram_prev or 0,
        close=last_close,
        bb_upper=indicators.bb_upper or last_close + 1,
        bb_lower=indicators.bb_lower or last_close - 1,
        bb_pct_b=indicators.bb_pct_b or 0.5,
        ema_9=indicators.ema_9 or last_close,
        ema_20=indicators.ema_20 or last_close,
        ema_50=indicators.ema_50 or last_close,
        adx=indicators.adx or 0,
        relative_volume=rel_vol,
        is_bullish_candle=is_bull,
    )

    # Structure
    deep_structure = MarketStructureAnalyzer().analyze(df)

    # HTF bias
    htf_bias_str = "neutral"
    if htf_structure is not None:
        trend_val = getattr(htf_structure, "trend", None)
        if trend_val is not None:
            tv = trend_val.value if hasattr(trend_val, "value") else str(trend_val)
            if "up" in tv.lower():
                htf_bias_str = "bullish"
            elif "down" in tv.lower():
                htf_bias_str = "bearish"

    # Candle counts
    candle_bull = sum(1 for p in pattern_results if getattr(p, "bias", "") == "bullish")
    candle_bear = sum(1 for p in pattern_results if getattr(p, "bias", "") == "bearish")

    # Regime check
    regime_val = regime.regime.value if hasattr(regime.regime, "value") else str(regime.regime)
    trending_regimes = {"trending_up", "trending_down", "breakout"}

    bias = BullBearBiasEngine().score(BiasInput(
        indicator_bullish=indicator_bias.bullish_score,
        indicator_bearish=indicator_bias.bearish_score,
        structure_bias=deep_structure.trend_bias,
        structure_strength=deep_structure.strength,
        candle_bullish_count=candle_bull,
        candle_bearish_count=candle_bear,
        candle_total=len(pattern_results),
        regime_supports_direction=regime_val.lower() in trending_regimes,
        volume_confirms=rel_vol > 1.2,
        htf_bias=htf_bias_str,
    ))

    allowed_action = None
    if bias.net_bias == "bullish":
        allowed_action = SignalAction.BUY
    elif bias.net_bias == "bearish":
        allowed_action = SignalAction.SELL

    return bias, allowed_action, htf_bias_str, indicator_bias, deep_structure


def apply_macro_filters(symbol, asset_class_value, regime, df):
    """
    Run macro filters.

    Returns (total_confidence_adj, should_block, block_reason).
    """
    from libs.analysis.macro.news import NewsImpactEngine
    from libs.analysis.macro.sentiment import SentimentFilter
    from libs.analysis.macro.global_risk import GlobalRiskEngine

    adj = 0.0

    news = NewsImpactEngine().assess(symbol, asset_class_value)
    if news.should_block:
        return 0.0, True, news.explanation
    adj += news.total_confidence_impact

    atr_pct = (
        (regime.atr / float(df["close"].iloc[-1]) * 100)
        if regime.atr and float(df["close"].iloc[-1]) > 0
        else 1.0
    )
    sentiment = SentimentFilter().assess(fear_greed_value=50, volatility_pct=atr_pct)
    adj += sentiment.confidence_adjustment

    global_risk = GlobalRiskEngine().assess()
    if global_risk.risk_level == "extreme":
        return 0.0, True, f"Global risk extreme: {global_risk.risk_score}"
    adj += global_risk.confidence_adjustment

    return adj, False, ""

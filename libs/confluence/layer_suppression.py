"""
Layer 5 — Risk Suppression Layer.

Detects conditions that REDUCE confidence or BLOCK entries:
  - Exhaustion signals
  - Divergence
  - Weak volume
  - Volatility collapse
  - Momentum decay
  - Overextension
  - Low liquidity
  - Fake breakout probability

Each suppression subtracts 5-15 points.
3+ suppressions → hard block regardless of score.
"""
from __future__ import annotations

from libs.analysis.regime.engine import RegimeAnalysis
from libs.analysis.structure.engine import MarketStructure


# ── Suppression definitions ──────────────────────────────────────────────────

_SUPPRESSION_PENALTIES: dict[str, float] = {
    "exhaustion":          -12.0,
    "rsi_divergence":      -10.0,
    "macd_divergence":      -8.0,
    "weak_volume":          -7.0,
    "volatility_collapse":  -8.0,
    "momentum_decay":       -6.0,
    "overextension":       -10.0,
    "low_liquidity":        -5.0,
    "fake_breakout_risk":   -9.0,
    "counter_trend":        -7.0,
}

MAX_SUPPRESSIONS_BEFORE_BLOCK = 3


def score_suppressions(
    indicators: dict | None,
    regime: RegimeAnalysis | None,
    structure: MarketStructure | None,
    direction: str,
    trigger_strategy: str = "",
) -> tuple[float, list[str], bool]:
    """Score the suppression layer.

    Args:
        indicators: Computed indicator values.
        regime: Regime analysis.
        structure: Market structure.
        direction: "BUY" or "SELL".
        trigger_strategy: Name of primary trigger (for breakout-specific checks).

    Returns:
        (penalty_total, suppression_signals, hard_blocked)
    """
    suppressions: list[str] = []
    total_penalty = 0.0

    if not indicators:
        return 0.0, [], False

    rsi = indicators.get("rsi")
    macd_hist = indicators.get("macd_histogram")
    rel_vol = indicators.get("volume_relative", 1.0)
    adx = indicators.get("adx")
    atr = indicators.get("atr")
    close = indicators.get("close")
    ema20 = indicators.get("ema_20")

    # ── RSI extreme / exhaustion ──
    if rsi is not None:
        if direction == "BUY" and rsi > 78:
            suppressions.append("exhaustion")
            total_penalty += _SUPPRESSION_PENALTIES["exhaustion"]
        elif direction == "SELL" and rsi < 22:
            suppressions.append("exhaustion")
            total_penalty += _SUPPRESSION_PENALTIES["exhaustion"]

    # ── RSI divergence (simplified) ──
    # BUY but RSI declining = bearish divergence
    if rsi is not None:
        prev_rsi = indicators.get("prev_rsi")
        if prev_rsi is not None:
            if direction == "BUY" and rsi < prev_rsi - 5 and rsi > 60:
                suppressions.append("rsi_divergence")
                total_penalty += _SUPPRESSION_PENALTIES["rsi_divergence"]
            elif direction == "SELL" and rsi > prev_rsi + 5 and rsi < 40:
                suppressions.append("rsi_divergence")
                total_penalty += _SUPPRESSION_PENALTIES["rsi_divergence"]

    # ── Weak volume ──
    if rel_vol is not None and rel_vol < 0.6:
        suppressions.append("weak_volume")
        total_penalty += _SUPPRESSION_PENALTIES["weak_volume"]

    # ── Momentum decay (ADX collapsing) ──
    if adx is not None and adx < 15:
        suppressions.append("momentum_decay")
        total_penalty += _SUPPRESSION_PENALTIES["momentum_decay"]

    # ── Overextension (price > 2 ATR from EMA20) ──
    if atr and close and ema20:
        distance = abs(close - ema20)
        if distance > 2.0 * atr:
            suppressions.append("overextension")
            total_penalty += _SUPPRESSION_PENALTIES["overextension"]

    # ── Volatility collapse ──
    if atr and close and atr / close < 0.001:
        suppressions.append("volatility_collapse")
        total_penalty += _SUPPRESSION_PENALTIES["volatility_collapse"]

    # ── Low liquidity regime ──
    if regime:
        regime_name = regime.name.lower() if hasattr(regime, "name") else ""
        if "low_liquidity" in regime_name:
            suppressions.append("low_liquidity")
            total_penalty += _SUPPRESSION_PENALTIES["low_liquidity"]

    # ── Counter-trend check ──
    if structure and structure.trend:
        from libs.core.models.domain import TrendDirection
        if direction == "BUY" and structure.trend == TrendDirection.DOWNTREND:
            if structure.trend_strength > 0.7:
                suppressions.append("counter_trend")
                total_penalty += _SUPPRESSION_PENALTIES["counter_trend"]
        elif direction == "SELL" and structure.trend == TrendDirection.UPTREND:
            if structure.trend_strength > 0.7:
                suppressions.append("counter_trend")
                total_penalty += _SUPPRESSION_PENALTIES["counter_trend"]

    # ── Fake breakout risk (breakout strategies in ranging regime) ──
    if regime and trigger_strategy:
        regime_name = regime.name.lower() if hasattr(regime, "name") else ""
        is_breakout = "breakout" in trigger_strategy
        is_ranging = "ranging" in regime_name
        if is_breakout and is_ranging:
            suppressions.append("fake_breakout_risk")
            total_penalty += _SUPPRESSION_PENALTIES["fake_breakout_risk"]

    hard_blocked = len(suppressions) >= MAX_SUPPRESSIONS_BEFORE_BLOCK
    return total_penalty, suppressions, hard_blocked

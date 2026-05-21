"""Agent 1: Market Regime Detection Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent

# Regime-to-bias mapping: positive = bullish-friendly, negative = bearish-friendly
_REGIME_BIAS: dict[str, float] = {
    "trending_up": 0.8,
    "trending_down": -0.8,
    "ranging_low_vol": 0.0,
    "ranging_high_vol": -0.1,
    "breakout": 0.3,
    "climactic": -0.5,
    "accumulation": 0.4,
    "distribution": -0.4,
    "panic_selloff": -0.9,
    "liquidation_event": -1.0,
    "reversal": 0.0,
    "low_liquidity": -0.3,
    "compression": 0.1,
    "expansion": 0.2,
    "news_driven": 0.0,
    "mean_reversion": 0.0,
    "unknown": 0.0,
}

# Regimes where trading is discouraged
_CAUTION_REGIMES = {
    "panic_selloff", "liquidation_event", "climactic",
    "low_liquidity", "news_driven",
}


class MarketRegimeAgent(TradingAgent):
    """Classifies market regime and assesses tradability."""

    @property
    def name(self) -> str:
        return "market_regime"

    @property
    def weight(self) -> float:
        return 1.5  # Regime is foundational — higher weight

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        regime = str(self._safe_get(context, "regime", "unknown"))
        vol_score = float(self._safe_get(context, "vol_score", 0.5))
        is_expanding = bool(self._safe_get(context, "is_expanding", False))
        atr_pct = float(self._safe_get(context, "atr_pct", 0.0))

        score = _REGIME_BIAS.get(regime, 0.0)

        # Adjust score by volatility conditions
        if is_expanding and regime not in _CAUTION_REGIMES:
            score *= 1.2

        score = self._clamp(score)

        # Confidence based on how clearly the regime is identified
        confidence = 0.6
        if regime in ("trending_up", "trending_down", "panic_selloff"):
            confidence = 0.85
        elif regime == "unknown":
            confidence = 0.2

        should_trade = regime not in _CAUTION_REGIMES
        risk_level = 0.8 if regime in _CAUTION_REGIMES else 0.3

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            reasoning=f"Regime={regime}, vol_score={vol_score:.2f}, atr_pct={atr_pct:.2f}",
            signals={"regime": regime, "vol_score": vol_score, "is_expanding": is_expanding},
            should_trade=should_trade,
            risk_level=risk_level,
        )

"""Agent 7: Strategy Selection Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent

# Which strategies work best in which regimes
_REGIME_STRATEGY_FIT: dict[str, list[str]] = {
    "trending_up": ["momentum", "trend", "breakout", "continuation"],
    "trending_down": ["momentum", "trend", "reversal"],
    "ranging_low_vol": ["mean_reversion", "reversal"],
    "ranging_high_vol": ["mean_reversion", "reversal"],
    "breakout": ["breakout", "momentum"],
    "climactic": [],  # no strategies recommended
    "accumulation": ["reversal", "continuation"],
    "distribution": ["reversal"],
    "panic_selloff": [],
    "liquidation_event": [],
    "reversal": ["reversal", "mean_reversion"],
    "low_liquidity": [],
    "compression": ["breakout"],
    "expansion": ["momentum", "breakout"],
    "news_driven": [],
    "mean_reversion": ["mean_reversion", "reversal"],
    "unknown": ["trend"],
}


class StrategySelectionAgent(TradingAgent):
    """Recommends optimal strategy based on regime and conditions."""

    @property
    def name(self) -> str:
        return "strategy_selection"

    @property
    def weight(self) -> float:
        return 1.2

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        regime = str(self._safe_get(context, "regime", "unknown"))
        proposed_strategy = str(self._safe_get(context, "strategy", ""))
        win_rates: dict[str, float] = self._safe_get(context, "strategy_win_rates", {}) or {}

        good_strategies = _REGIME_STRATEGY_FIT.get(regime, ["trend"])
        is_good_fit = proposed_strategy in good_strategies

        score = 0.3 if is_good_fit else -0.2
        reasons: list[str] = []

        if is_good_fit:
            reasons.append(f"{proposed_strategy} fits {regime} regime")
        elif good_strategies:
            reasons.append(f"{proposed_strategy} poor fit for {regime}; prefer {good_strategies[0]}")
        else:
            reasons.append(f"No strategies recommended for {regime}")
            score = -0.5

        # Win rate adjustment
        if proposed_strategy in win_rates:
            wr = win_rates[proposed_strategy]
            if wr > 0.6:
                score += 0.2
                reasons.append(f"Strong win rate ({wr:.0%})")
            elif wr < 0.4:
                score -= 0.2
                reasons.append(f"Weak win rate ({wr:.0%})")

        score = self._clamp(score)
        confidence = 0.7 if good_strategies else 0.3

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            reasoning="; ".join(reasons),
            signals={
                "proposed": proposed_strategy,
                "recommended": good_strategies,
                "is_good_fit": is_good_fit,
            },
            should_trade=bool(good_strategies) and is_good_fit,
            risk_level=0.3 if is_good_fit else 0.6,
        )

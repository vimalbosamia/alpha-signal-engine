"""Agent 11: Execution Optimization Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class ExecutionOptimizationAgent(TradingAgent):
    """Optimizes execution: timing, spread, slippage, order type selection."""

    @property
    def name(self) -> str:
        return "execution_optimization"

    @property
    def weight(self) -> float:
        return 0.8

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        spread_pct = float(self._safe_get(context, "spread_pct", 0.0))
        expected_slippage_pct = float(self._safe_get(context, "expected_slippage_pct", 0.0))
        volume_ratio = float(self._safe_get(context, "volume_ratio", 1.0))
        session = str(self._safe_get(context, "session", "continuous"))
        atr_pct = float(self._safe_get(context, "atr_pct", 0.0))
        order_book_depth = float(self._safe_get(context, "order_book_depth", 0.0))

        score = 0.0
        reasons: list[str] = []
        risk_level = 0.3
        signals: dict[str, Any] = {}

        # Spread cost analysis
        total_cost_pct = spread_pct + expected_slippage_pct
        if total_cost_pct > 0.5:
            score -= 0.3
            reasons.append(f"High execution cost ({total_cost_pct:.2f}%)")
            risk_level = 0.7
        elif total_cost_pct > 0.2:
            score -= 0.1
            reasons.append(f"Moderate execution cost ({total_cost_pct:.2f}%)")

        # Liquidity check
        if volume_ratio > 1.5:
            score += 0.2
            reasons.append("Good liquidity for execution")
        elif volume_ratio < 0.3:
            score -= 0.3
            reasons.append("Poor liquidity — execution risk")
            risk_level = max(risk_level, 0.7)

        # Order type recommendation
        if spread_pct > 0.3 or volume_ratio < 0.5:
            signals["recommended_order_type"] = "limit"
            signals["use_ladder_entry"] = True
            reasons.append("Recommend limit orders with ladder entry")
        elif atr_pct > 3.0:
            signals["recommended_order_type"] = "limit"
            signals["use_ladder_entry"] = False
            reasons.append("Recommend limit orders (volatile)")
        else:
            signals["recommended_order_type"] = "market"
            signals["use_ladder_entry"] = False

        # Session timing
        if session in ("pre_market", "post_market"):
            score -= 0.1
            reasons.append("Off-hours session — wider spreads")
            risk_level = max(risk_level, 0.5)

        # Trailing stop recommendation based on ATR
        if atr_pct > 0:
            signals["recommended_trailing_stop_pct"] = round(atr_pct * 1.5, 2)

        signals["total_cost_pct"] = total_cost_pct
        signals["execution_score"] = max(0, 1.0 - total_cost_pct * 2)

        score = self._clamp(score)
        should_trade = total_cost_pct < 1.0 and volume_ratio > 0.2

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=0.7,
            reasoning="; ".join(reasons) or "Good execution conditions",
            signals=signals,
            should_trade=should_trade,
            risk_level=risk_level,
        )

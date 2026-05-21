"""Agent 14: Performance Analytics Agent."""
from __future__ import annotations

from typing import Any

from libs.agents.base import AgentOutput, TradingAgent


class PerformanceAnalyticsAgent(TradingAgent):
    """Tracks system performance: Sharpe, drawdown trends, strategy degradation."""

    @property
    def name(self) -> str:
        return "performance_analytics"

    @property
    def weight(self) -> float:
        return 0.9

    def analyze(self, context: dict[str, Any]) -> AgentOutput:
        total_trades = int(self._safe_get(context, "total_trades", 0))
        win_rate = float(self._safe_get(context, "win_rate", 0.5))
        sharpe_ratio = float(self._safe_get(context, "sharpe_ratio", 0.0))
        max_drawdown_pct = float(self._safe_get(context, "max_drawdown_pct", 0.0))
        recent_win_rate = float(self._safe_get(context, "recent_win_rate", 0.5))
        avg_rr = float(self._safe_get(context, "avg_rr", 1.0))

        score = 0.0
        reasons: list[str] = []

        if total_trades < 10:
            return AgentOutput(
                agent_name=self.name,
                score=0.0,
                confidence=0.15,
                reasoning=f"Insufficient trade history ({total_trades} trades)",
                should_trade=True,
                risk_level=0.5,
            )

        # Sharpe ratio assessment
        if sharpe_ratio > 2.0:
            score += 0.3
            reasons.append(f"Excellent Sharpe ({sharpe_ratio:.2f})")
        elif sharpe_ratio > 1.0:
            score += 0.15
            reasons.append(f"Good Sharpe ({sharpe_ratio:.2f})")
        elif sharpe_ratio < 0:
            score -= 0.3
            reasons.append(f"Negative Sharpe ({sharpe_ratio:.2f})")

        # Win rate trend (recent vs overall)
        trend_diff = recent_win_rate - win_rate
        if trend_diff < -0.1:
            score -= 0.3
            reasons.append(f"Degrading win rate ({recent_win_rate:.0%} vs {win_rate:.0%})")
        elif trend_diff > 0.1:
            score += 0.2
            reasons.append(f"Improving win rate ({recent_win_rate:.0%} vs {win_rate:.0%})")

        # Average RR
        if avg_rr > 2.0:
            score += 0.2
            reasons.append(f"Strong avg R:R ({avg_rr:.1f})")
        elif avg_rr < 1.0:
            score -= 0.2
            reasons.append(f"Weak avg R:R ({avg_rr:.1f})")

        # Drawdown trend
        if max_drawdown_pct > 15:
            score -= 0.3
            reasons.append(f"Critical drawdown ({max_drawdown_pct:.1f}%)")

        score = self._clamp(score)
        confidence = min(0.8, 0.3 + total_trades / 200)
        should_trade = recent_win_rate > 0.35 and max_drawdown_pct < 20

        return AgentOutput(
            agent_name=self.name,
            score=score,
            confidence=round(confidence, 3),
            reasoning="; ".join(reasons),
            signals={
                "sharpe": sharpe_ratio,
                "win_rate": win_rate,
                "recent_win_rate": recent_win_rate,
                "avg_rr": avg_rr,
                "total_trades": total_trades,
            },
            should_trade=should_trade,
            risk_level=0.3 if sharpe_ratio > 1.0 else 0.6,
        )

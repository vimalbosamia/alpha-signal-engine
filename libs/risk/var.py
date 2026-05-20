"""
Value at Risk (VaR) engine using historical simulation.

Computes VaR_95, VaR_99, and CVaR_95 (expected shortfall) from a sequence
of daily percentage returns.  No external dependencies required.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VaRResult:
    var_95: float          # 95% VaR — absolute loss in currency units
    var_99: float          # 99% VaR
    cvar_95: float         # Conditional VaR (expected shortfall beyond VaR_95)
    daily_risk_pct: float  # VaR_95 as a percentage of portfolio value
    is_acceptable: bool    # True if daily_risk_pct < 3%
    explanation: str


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the p-th percentile of a pre-sorted list (0 <= pct <= 1)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = pct * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


class VaREngine:
    """
    Stateless historical Value at Risk calculator.

    Instantiate once; call calculate() per portfolio / time window.
    """

    _ACCEPTABLE_DAILY_RISK_PCT = 3.0

    def calculate(
        self,
        daily_returns: list[float],
        portfolio_value: float = 10_000.0,
        confidence_95: float = 0.05,
        confidence_99: float = 0.01,
    ) -> VaRResult:
        """
        Historical VaR: sort daily returns and find percentile cutoffs.

        VaR_95  = 5th percentile of returns * portfolio_value  (a loss, so negative)
        CVaR_95 = mean of returns that fall at or below the VaR_95 threshold

        Args:
            daily_returns:    Daily percentage returns (e.g. -1.5 means -1.5%).
            portfolio_value:  Current portfolio value in currency units.
            confidence_95:    Lower-tail probability for 95% VaR (default 0.05).
            confidence_99:    Lower-tail probability for 99% VaR (default 0.01).

        Returns:
            VaRResult with risk metrics and acceptability flag.
        """
        if not daily_returns:
            return VaRResult(
                var_95=0.0,
                var_99=0.0,
                cvar_95=0.0,
                daily_risk_pct=0.0,
                is_acceptable=True,
                explanation="No daily returns provided; VaR is zero by default.",
            )

        sorted_returns = sorted(daily_returns)

        # Percentile cutoffs (returns are in %)
        p5_return = _percentile(sorted_returns, confidence_95)
        p1_return = _percentile(sorted_returns, confidence_99)

        # VaR expressed as a positive loss value in currency units
        var_95 = abs(p5_return / 100.0 * portfolio_value)
        var_99 = abs(p1_return / 100.0 * portfolio_value)

        # CVaR_95: expected loss given that we are beyond the 95% VaR threshold
        cutoff_return = p5_return
        tail_returns = [r for r in sorted_returns if r <= cutoff_return]
        if tail_returns:
            mean_tail_return = sum(tail_returns) / len(tail_returns)
        else:
            mean_tail_return = p5_return
        cvar_95 = abs(mean_tail_return / 100.0 * portfolio_value)

        daily_risk_pct = (var_95 / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0
        is_acceptable = daily_risk_pct < self._ACCEPTABLE_DAILY_RISK_PCT

        explanation = (
            f"Historical VaR on {len(daily_returns)} daily returns "
            f"(portfolio: ${portfolio_value:,.0f}). "
            f"VaR_95: ${var_95:.2f} ({daily_risk_pct:.2f}% of portfolio), "
            f"VaR_99: ${var_99:.2f}, "
            f"CVaR_95: ${cvar_95:.2f}. "
            f"Daily risk is {'ACCEPTABLE' if is_acceptable else 'HIGH'} "
            f"(threshold: {self._ACCEPTABLE_DAILY_RISK_PCT}%)."
        )

        return VaRResult(
            var_95=round(var_95, 4),
            var_99=round(var_99, 4),
            cvar_95=round(cvar_95, 4),
            daily_risk_pct=round(daily_risk_pct, 4),
            is_acceptable=is_acceptable,
            explanation=explanation,
        )

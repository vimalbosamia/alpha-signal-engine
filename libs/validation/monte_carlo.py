"""
Monte Carlo simulation engine for validating strategy robustness.

Randomly resamples from historical trade returns to simulate many possible
outcome paths. No numpy required — uses only stdlib random.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MonteCarloResult:
    simulations: int
    median_return_pct: float
    p5_return_pct: float       # 5th percentile (worst case)
    p95_return_pct: float      # 95th percentile (best case)
    probability_of_profit: float  # fraction of sims that end profitable
    max_drawdown_median: float
    is_robust: bool            # True if >60% profitable AND median > 0
    explanation: str


def _cumulative_return(returns: list[float]) -> float:
    """Compound a sequence of percentage returns into a single total return."""
    result = 1.0
    for r in returns:
        result *= 1.0 + r / 100.0
    return (result - 1.0) * 100.0


def _max_drawdown(returns: list[float]) -> float:
    """Compute maximum peak-to-trough drawdown for a sequence of returns."""
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= 1.0 + r / 100.0
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


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


class MonteCarloEngine:
    """
    Validates a strategy's robustness via bootstrap resampling.

    Instantiate once; call simulate() per strategy under evaluation.
    """

    def simulate(
        self,
        trade_returns: list[float],
        n_simulations: int = 1000,
        n_trades_per_sim: int = 100,
    ) -> MonteCarloResult:
        """
        Randomly resample from historical trade returns to simulate many
        possible outcomes.

        Each simulation picks n_trades_per_sim returns randomly with
        replacement, calculates cumulative return and max drawdown.

        Args:
            trade_returns:     Historical per-trade returns as percentages
                               (e.g. 2.5 means +2.5%).
            n_simulations:     Number of Monte Carlo paths to run.
            n_trades_per_sim:  Number of trades to sample per simulation path.

        Returns:
            MonteCarloResult with cross-simulation statistics.
        """
        if not trade_returns:
            return MonteCarloResult(
                simulations=0,
                median_return_pct=0.0,
                p5_return_pct=0.0,
                p95_return_pct=0.0,
                probability_of_profit=0.0,
                max_drawdown_median=0.0,
                is_robust=False,
                explanation="No trade returns provided; cannot simulate.",
            )

        final_returns: list[float] = []
        drawdowns: list[float] = []

        for _ in range(n_simulations):
            sample = random.choices(trade_returns, k=n_trades_per_sim)
            final_returns.append(_cumulative_return(sample))
            drawdowns.append(_max_drawdown(sample))

        final_returns.sort()
        drawdowns.sort()

        median_return = _percentile(final_returns, 0.50)
        p5_return = _percentile(final_returns, 0.05)
        p95_return = _percentile(final_returns, 0.95)
        dd_median = _percentile(drawdowns, 0.50)

        n_profitable = sum(1 for r in final_returns if r > 0)
        prob_profit = n_profitable / n_simulations

        is_robust = prob_profit > 0.60 and median_return > 0

        explanation = (
            f"Ran {n_simulations} simulations of {n_trades_per_sim} trades each. "
            f"Median return: {median_return:.2f}%, "
            f"5th–95th pct range: [{p5_return:.2f}%, {p95_return:.2f}%], "
            f"Probability of profit: {prob_profit * 100:.1f}%, "
            f"Median max drawdown: {dd_median:.2f}%. "
            f"Strategy is {'ROBUST' if is_robust else 'NOT ROBUST'}."
        )

        return MonteCarloResult(
            simulations=n_simulations,
            median_return_pct=round(median_return, 4),
            p5_return_pct=round(p5_return, 4),
            p95_return_pct=round(p95_return, 4),
            probability_of_profit=round(prob_profit, 4),
            max_drawdown_median=round(dd_median, 4),
            is_robust=is_robust,
            explanation=explanation,
        )

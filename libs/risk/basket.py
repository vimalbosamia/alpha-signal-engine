"""
Basket / Portfolio Expectancy Engine.

Analyzes multiple open signals as a basket:
  - Calculates expected return at various win rates
  - Detects concentration risk (asset class skew)
  - Detects correlation risk (duplicate symbol+direction)
  - Produces actionable warnings

This module is purely analytical — no trade execution occurs here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# ── Win rates at which expected return is evaluated ───────────────────────────
_WIN_RATE_LABELS: list[tuple[str, float]] = [
    ("40%", 0.40),
    ("50%", 0.50),
    ("55%", 0.55),
    ("60%", 0.60),
    ("65%", 0.65),
    ("70%", 0.70),
]

# Thresholds for concentration risk
_HIGH_CONCENTRATION_THRESHOLD: float = 70.0   # % of signals in one asset class
_MODERATE_CONCENTRATION_THRESHOLD: float = 60.0
_HIGH_DUPLICATE_THRESHOLD: int = 3            # same symbol+direction count

# R:R below which a "low R:R" warning is emitted
_LOW_RR_THRESHOLD: float = 1.5

# % crypto that triggers a "too many crypto longs/shorts" warning
_CRYPTO_DIRECTIONAL_WARNING_THRESHOLD: float = 60.0


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BasketSignal:
    """One signal in the basket."""

    symbol: str
    asset_class: str    # "crypto" or "stock"
    action: str         # "BUY" or "SELL"
    tp1_pct: float      # potential gain to TP1 as %
    stop_pct: float     # potential loss to stop as %
    risk_reward: float
    confidence: float


@dataclass(frozen=True)
class BasketAnalysis:
    """Aggregate analysis of all signals in the basket."""

    signal_count: int
    buy_count: int
    sell_count: int
    avg_risk_reward: float
    avg_tp1_pct: float
    avg_stop_pct: float
    worst_case_loss_pct: float    # if ALL stops hit (negative value)
    best_case_gain_pct: float     # if ALL TP1s hit
    breakeven_win_rate: float     # win rate needed to break even (0–1)
    expected_returns: dict[str, float]  # {"40%": X, "50%": X, …}
    crypto_exposure_pct: float    # % of signals that are crypto
    stock_exposure_pct: float     # % of signals that are stock
    duplicate_direction_count: int  # number of distinct symbol+direction pairs with >1 occurrence
    concentration_risk: str       # "low", "moderate", or "high"
    warnings: list[str]


# ── Engine ────────────────────────────────────────────────────────────────────

class BasketExpectancyEngine:
    """
    Stateless basket analyser.

    Instantiate once; call analyze() per basket.
    """

    def analyze(self, signals: list[BasketSignal]) -> BasketAnalysis:
        """Analyze a basket of signals for expected return and risk."""
        if not signals:
            return self._empty_analysis()

        total = len(signals)

        # Directional counts
        buy_count = sum(1 for s in signals if s.action.upper() == "BUY")
        sell_count = total - buy_count

        # Averages
        avg_rr = sum(s.risk_reward for s in signals) / total
        avg_tp1 = sum(s.tp1_pct for s in signals) / total
        avg_stop = sum(s.stop_pct for s in signals) / total

        # Aggregate scenario values
        worst_case = -sum(s.stop_pct for s in signals)
        best_case = sum(s.tp1_pct for s in signals)

        # Breakeven win rate: EV = 0 when wr * avg_tp1 = (1 - wr) * avg_stop
        # → wr = avg_stop / (avg_tp1 + avg_stop)
        # Equivalent to: 1 / (1 + avg_rr)
        breakeven_wr = 1.0 / (1.0 + avg_rr) if avg_rr > 0 else 0.0

        # Expected returns at various win rates
        expected_returns = {
            label: self._expected_return(wr, avg_tp1, avg_stop)
            for label, wr in _WIN_RATE_LABELS
        }

        # Asset class exposure
        crypto_count = sum(1 for s in signals if s.asset_class.lower() == "crypto")
        stock_count = total - crypto_count
        crypto_pct = crypto_count / total * 100.0
        stock_pct = stock_count / total * 100.0

        # Duplicate direction detection
        direction_counter = Counter(
            (s.symbol.upper(), s.action.upper()) for s in signals
        )
        dup_pairs = {pair for pair, cnt in direction_counter.items() if cnt > 1}
        duplicate_direction_count = len(dup_pairs)

        # Concentration risk
        concentration_risk = self._concentration_risk(
            crypto_pct=crypto_pct,
            direction_counter=direction_counter,
        )

        # Warnings
        warnings = self._build_warnings(
            signals=signals,
            crypto_pct=crypto_pct,
            buy_count=buy_count,
            sell_count=sell_count,
            avg_rr=avg_rr,
            direction_counter=direction_counter,
        )

        return BasketAnalysis(
            signal_count=total,
            buy_count=buy_count,
            sell_count=sell_count,
            avg_risk_reward=avg_rr,
            avg_tp1_pct=avg_tp1,
            avg_stop_pct=avg_stop,
            worst_case_loss_pct=worst_case,
            best_case_gain_pct=best_case,
            breakeven_win_rate=breakeven_wr,
            expected_returns=expected_returns,
            crypto_exposure_pct=crypto_pct,
            stock_exposure_pct=stock_pct,
            duplicate_direction_count=duplicate_direction_count,
            concentration_risk=concentration_risk,
            warnings=warnings,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _expected_return(wr: float, avg_tp1: float, avg_stop: float) -> float:
        """
        Expected return per unit of capital at win-rate *wr*.

        Formula: (wr * avg_tp1) - ((1 - wr) * avg_stop)
        """
        return (wr * avg_tp1) - ((1.0 - wr) * avg_stop)

    @staticmethod
    def _concentration_risk(
        crypto_pct: float,
        direction_counter: Counter,
    ) -> str:
        """
        Classify concentration risk as "low", "moderate", or "high".

        "high" triggers when:
          - >70% of signals are in one asset class, OR
          - any symbol+direction pair has >=3 occurrences
        "moderate": >60% in one asset class
        "low" otherwise
        """
        stock_pct = 100.0 - crypto_pct

        dominant_pct = max(crypto_pct, stock_pct)

        # High: heavy asset class skew or heavy directional repetition
        max_pair_count = max(direction_counter.values(), default=0)
        if dominant_pct >= _HIGH_CONCENTRATION_THRESHOLD or max_pair_count >= _HIGH_DUPLICATE_THRESHOLD:
            return "high"

        if dominant_pct > _MODERATE_CONCENTRATION_THRESHOLD:
            return "moderate"

        return "low"

    @staticmethod
    def _build_warnings(
        signals: list[BasketSignal],
        crypto_pct: float,
        buy_count: int,
        sell_count: int,
        avg_rr: float,
        direction_counter: Counter,
    ) -> list[str]:
        """Produce human-readable warnings about basket composition."""
        warnings: list[str] = []
        total = len(signals)

        # Low average R:R
        if avg_rr < _LOW_RR_THRESHOLD:
            warnings.append(
                f"Low avg R:R ({avg_rr:.2f}) — basket edge may be insufficient. "
                f"Aim for at least {_LOW_RR_THRESHOLD}."
            )

        # Too many crypto longs
        crypto_buy_count = sum(
            1 for s in signals
            if s.asset_class.lower() == "crypto" and s.action.upper() == "BUY"
        )
        if total > 0 and (crypto_buy_count / total * 100.0) >= _CRYPTO_DIRECTIONAL_WARNING_THRESHOLD:
            warnings.append(
                f"Too many crypto longs ({crypto_buy_count}/{total} signals) — "
                "correlated downside risk if BTC sells off."
            )

        # Too many crypto shorts
        crypto_sell_count = sum(
            1 for s in signals
            if s.asset_class.lower() == "crypto" and s.action.upper() == "SELL"
        )
        if total > 0 and (crypto_sell_count / total * 100.0) >= _CRYPTO_DIRECTIONAL_WARNING_THRESHOLD:
            warnings.append(
                f"Too many crypto shorts ({crypto_sell_count}/{total} signals) — "
                "correlated upside risk if BTC rallies."
            )

        # Same symbol appearing multiple times with the same direction
        dup_symbols = [
            f"{symbol}({action})"
            for (symbol, action), cnt in direction_counter.items()
            if cnt > 1
        ]
        if dup_symbols:
            warnings.append(
                f"Same symbol and direction appear multiple times: "
                f"{', '.join(sorted(dup_symbols))}. "
                "This inflates effective exposure."
            )

        return warnings

    @staticmethod
    def _empty_analysis() -> BasketAnalysis:
        """Return a safe neutral analysis for an empty basket."""
        return BasketAnalysis(
            signal_count=0,
            buy_count=0,
            sell_count=0,
            avg_risk_reward=0.0,
            avg_tp1_pct=0.0,
            avg_stop_pct=0.0,
            worst_case_loss_pct=0.0,
            best_case_gain_pct=0.0,
            breakeven_win_rate=0.0,
            expected_returns={label: 0.0 for label, _ in _WIN_RATE_LABELS},
            crypto_exposure_pct=0.0,
            stock_exposure_pct=0.0,
            duplicate_direction_count=0,
            concentration_risk="low",
            warnings=[],
        )

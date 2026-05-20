"""
libs.risk.cost_model — Trading cost estimation engine.

Estimates round-trip trading costs (spread, slippage, commission) for a given
symbol and asset class, then determines whether a trade is viable given the
expected price move.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Cost tables ───────────────────────────────────────────────────────────────
# All values are per-side percentages (applied to entry price).

_CRYPTO_SPOT_SPREAD_PCT = 0.05
_CRYPTO_SPOT_SLIPPAGE_PCT = 0.03
_CRYPTO_SPOT_COMMISSION_PCT = 0.10

_CRYPTO_FUTURES_SPREAD_PCT = 0.02
_CRYPTO_FUTURES_SLIPPAGE_PCT = 0.02
_CRYPTO_FUTURES_COMMISSION_PCT = 0.04

_STOCK_SPREAD_PCT = 0.01
_STOCK_SLIPPAGE_PCT = 0.01
_STOCK_COMMISSION_PCT = 0.00  # zero-commission brokers

# Small-cap / low-liquidity spread multiplier
_LOW_LIQUIDITY_SPREAD_MULTIPLIER = 2.0

# Round-trip multiplier (enter + exit)
_ROUND_TRIP = 2.0

# Minimum profitable move safety factor (costs * factor)
_MIN_PROFITABLE_FACTOR = 1.5

# Asset class identifiers (normalised to lower-case for matching)
_ASSET_CRYPTO = "crypto"
_ASSET_STOCK = "stock"
_ASSET_EQUITY = "equity"

# Symbols typically considered small-cap / low-liquidity
_SMALL_CAP_INDICATORS = {"small_cap", "low_liquidity", "otc", "penny"}


@dataclass(frozen=True)
class TradingCosts:
    spread_pct: float
    slippage_pct: float
    commission_pct: float
    total_cost_pct: float
    min_profitable_move_pct: float
    should_reject: bool
    explanation: str


class CostModelEngine:
    """
    Estimates round-trip trading costs and determines trade viability.

    Cost schedule (per side):
        Crypto spot:    spread 0.05%, slippage 0.03%, commission 0.10%
        Crypto futures: spread 0.02%, slippage 0.02%, commission 0.04%
        Stock/equity:   spread 0.01%, slippage 0.01%, commission 0%
        Small-cap:      spread × 2

    Round-trip total = (spread + slippage + commission) × 2
    Min profitable move = total_cost × 1.5
    Reject when expected_move_pct < min_profitable_move_pct
    """

    def assess(
        self,
        symbol: str,
        asset_class: str,
        entry_price: float,
        expected_move_pct: float,
        position_size_usd: float = 1000,
        is_futures: bool = False,
    ) -> TradingCosts:
        """
        Compute cost structure and return a TradingCosts dataclass.

        Parameters
        ----------
        symbol:
            Ticker symbol (used for low-liquidity detection).
        asset_class:
            One of "crypto", "stock", or "equity".
        entry_price:
            Current price of the asset (used for context only; costs are %).
        expected_move_pct:
            Expected price move as a percentage (e.g. 1.5 for 1.5%).
        position_size_usd:
            Notional position size in USD (informational; not used in cost %).
        is_futures:
            Override to use futures cost schedule even when asset_class="crypto".
        """
        spread, slippage, commission = self._base_costs(asset_class, is_futures)

        if self._is_low_liquidity(symbol, asset_class):
            spread *= _LOW_LIQUIDITY_SPREAD_MULTIPLIER

        per_side = spread + slippage + commission
        total_rt = per_side * _ROUND_TRIP
        min_move = total_rt * _MIN_PROFITABLE_FACTOR

        should_reject = expected_move_pct < min_move
        explanation = self._build_explanation(
            spread, slippage, commission, total_rt, min_move,
            expected_move_pct, should_reject
        )

        return TradingCosts(
            spread_pct=round(spread, 6),
            slippage_pct=round(slippage, 6),
            commission_pct=round(commission, 6),
            total_cost_pct=round(total_rt, 6),
            min_profitable_move_pct=round(min_move, 6),
            should_reject=should_reject,
            explanation=explanation,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _base_costs(asset_class: str, is_futures: bool) -> tuple[float, float, float]:
        """Return (spread_pct, slippage_pct, commission_pct) per side."""
        ac = asset_class.lower().strip()
        if ac == _ASSET_CRYPTO:
            if is_futures:
                return (
                    _CRYPTO_FUTURES_SPREAD_PCT,
                    _CRYPTO_FUTURES_SLIPPAGE_PCT,
                    _CRYPTO_FUTURES_COMMISSION_PCT,
                )
            return (
                _CRYPTO_SPOT_SPREAD_PCT,
                _CRYPTO_SPOT_SLIPPAGE_PCT,
                _CRYPTO_SPOT_COMMISSION_PCT,
            )
        if ac in (_ASSET_STOCK, _ASSET_EQUITY):
            return (
                _STOCK_SPREAD_PCT,
                _STOCK_SLIPPAGE_PCT,
                _STOCK_COMMISSION_PCT,
            )
        # Default to crypto spot for unknown asset classes
        return (
            _CRYPTO_SPOT_SPREAD_PCT,
            _CRYPTO_SPOT_SLIPPAGE_PCT,
            _CRYPTO_SPOT_COMMISSION_PCT,
        )

    @staticmethod
    def _is_low_liquidity(symbol: str, asset_class: str) -> bool:
        """Heuristic: flag symbols that look like small-cap / low-liquidity."""
        combined = f"{symbol} {asset_class}".lower()
        return any(ind in combined for ind in _SMALL_CAP_INDICATORS)

    @staticmethod
    def _build_explanation(
        spread: float,
        slippage: float,
        commission: float,
        total_rt: float,
        min_move: float,
        expected_move: float,
        rejected: bool,
    ) -> str:
        verdict = "REJECTED" if rejected else "ACCEPTED"
        return (
            f"[{verdict}] spread={spread:.4f}% slippage={slippage:.4f}% "
            f"commission={commission:.4f}% per side | "
            f"round-trip total={total_rt:.4f}% | "
            f"min profitable move={min_move:.4f}% | "
            f"expected move={expected_move:.4f}%"
        )

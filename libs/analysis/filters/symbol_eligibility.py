"""
SymbolEligibilityEngine — pre-trade symbol quality gate.

Checks volume, spread, and volatility before a symbol enters
the signal pipeline.  All thresholds are configurable at init.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolEligibility:
    symbol: str
    is_eligible: bool
    reason: str
    volume_24h: float
    spread_quality: str   # "good" | "fair" | "poor"
    volatility_ok: bool
    liquidity_ok: bool


def _spread_quality(spread_pct: float) -> str:
    if spread_pct < 0.1:
        return "good"
    if spread_pct < 0.3:
        return "fair"
    return "poor"


class SymbolEligibilityEngine:
    """Gate symbols based on volume, spread, and volatility thresholds."""

    def __init__(
        self,
        min_volume_usd: float = 5_000_000,
        max_spread_pct: float = 0.5,
        min_atr_pct: float = 0.1,
        max_atr_pct: float = 20.0,
    ) -> None:
        self._min_volume = min_volume_usd
        self._max_spread = max_spread_pct
        self._min_atr = min_atr_pct
        self._max_atr = max_atr_pct

    def check(
        self,
        symbol: str,
        volume_24h: float,
        spread_pct: float,
        atr_pct: float,
    ) -> SymbolEligibility:
        """
        Rules (evaluated in order — first failure wins):
          - volume_24h < min_volume  → ineligible ("low volume")
          - spread_pct > max_spread  → ineligible ("wide spread")
          - atr_pct   < min_atr      → ineligible ("dead market")
          - atr_pct   > max_atr      → ineligible ("too volatile")
          - all pass                 → eligible
        """
        sq = _spread_quality(spread_pct)
        liquidity_ok = volume_24h >= self._min_volume and spread_pct <= self._max_spread
        volatility_ok = self._min_atr <= atr_pct <= self._max_atr

        if volume_24h < self._min_volume:
            return SymbolEligibility(
                symbol=symbol,
                is_eligible=False,
                reason="low volume",
                volume_24h=volume_24h,
                spread_quality=sq,
                volatility_ok=volatility_ok,
                liquidity_ok=False,
            )

        if spread_pct > self._max_spread:
            return SymbolEligibility(
                symbol=symbol,
                is_eligible=False,
                reason="wide spread",
                volume_24h=volume_24h,
                spread_quality=sq,
                volatility_ok=volatility_ok,
                liquidity_ok=False,
            )

        if atr_pct < self._min_atr:
            return SymbolEligibility(
                symbol=symbol,
                is_eligible=False,
                reason="dead market",
                volume_24h=volume_24h,
                spread_quality=sq,
                volatility_ok=False,
                liquidity_ok=liquidity_ok,
            )

        if atr_pct > self._max_atr:
            return SymbolEligibility(
                symbol=symbol,
                is_eligible=False,
                reason="too volatile",
                volume_24h=volume_24h,
                spread_quality=sq,
                volatility_ok=False,
                liquidity_ok=liquidity_ok,
            )

        return SymbolEligibility(
            symbol=symbol,
            is_eligible=True,
            reason="ok",
            volume_24h=volume_24h,
            spread_quality=sq,
            volatility_ok=True,
            liquidity_ok=True,
        )

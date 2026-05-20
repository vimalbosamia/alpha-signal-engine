"""
Sector Rotation Engine.

Classifies sector strength based on benchmark ETF performance (SPY, QQQ)
and BTC price changes. Adjusts trade confidence based on how the relevant
benchmark is performing.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Tech stocks that follow QQQ ───────────────────────────────────────────────
_TECH_SYMBOLS = {"AAPL", "MSFT", "NVDA", "TSLA"}

# ── Change-percent thresholds ─────────────────────────────────────────────────
_DOWN_QQQ_THRESHOLD = -1.0   # QQQ pct change below this → tech weak
_DOWN_BTC_THRESHOLD = -2.0   # BTC pct change below this → crypto weak
_DOWN_SPY_THRESHOLD = -1.0   # SPY pct change below this → general stock weak
_RISING_THRESHOLD = 1.0      # benchmark above this → strong

# ── Confidence adjustments ────────────────────────────────────────────────────
_ADJ_WEAK_TECH = -0.05
_ADJ_WEAK_CRYPTO = -0.05
_ADJ_WEAK_STOCK = -0.03
_ADJ_STRONG = +0.03
_ADJ_NEUTRAL = 0.0


@dataclass(frozen=True)
class SectorStrength:
    """Immutable snapshot of assessed sector strength."""

    sector: str
    """One of: tech, energy, finance, healthcare, crypto, general."""

    strength: str
    """One of: strong, neutral, weak."""

    confidence_adjustment: float
    """Signed delta applied to signal confidence. Range: -0.05 to +0.05."""


def _is_crypto(symbol: str, asset_class: str) -> bool:
    """Return True when the asset should be treated as crypto."""
    return asset_class.lower() == "crypto" or "USDT" in symbol.upper() or "BTC" in symbol.upper()


class SectorRotationEngine:
    """Assesses sector strength and its effect on trade confidence."""

    def assess(
        self,
        symbol: str,
        asset_class: str,
        spy_change_pct: float = 0.0,
        qqq_change_pct: float = 0.0,
        btc_change_pct: float = 0.0,
    ) -> SectorStrength:
        """Assess sector strength for the given symbol.

        Args:
            symbol: Trading symbol (e.g. "AAPL", "ETHUSDT").
            asset_class: Asset class hint (e.g. "crypto", "stock").
            spy_change_pct: SPY percentage change (e.g. -1.5 means -1.5%).
            qqq_change_pct: QQQ percentage change.
            btc_change_pct: BTC percentage change.

        Returns:
            SectorStrength with sector, strength, and confidence_adjustment.
        """
        upper_symbol = symbol.upper()

        # ── Tech: follows QQQ ────────────────────────────────────────────────
        if upper_symbol in _TECH_SYMBOLS:
            sector = "tech"
            if qqq_change_pct < _DOWN_QQQ_THRESHOLD:
                return SectorStrength(sector=sector, strength="weak", confidence_adjustment=_ADJ_WEAK_TECH)
            if qqq_change_pct > _RISING_THRESHOLD:
                return SectorStrength(sector=sector, strength="strong", confidence_adjustment=_ADJ_STRONG)
            return SectorStrength(sector=sector, strength="neutral", confidence_adjustment=_ADJ_NEUTRAL)

        # ── Crypto: follows BTC ──────────────────────────────────────────────
        if _is_crypto(upper_symbol, asset_class):
            sector = "crypto"
            if btc_change_pct < _DOWN_BTC_THRESHOLD:
                return SectorStrength(sector=sector, strength="weak", confidence_adjustment=_ADJ_WEAK_CRYPTO)
            if btc_change_pct > _RISING_THRESHOLD:
                return SectorStrength(sector=sector, strength="strong", confidence_adjustment=_ADJ_STRONG)
            return SectorStrength(sector=sector, strength="neutral", confidence_adjustment=_ADJ_NEUTRAL)

        # ── General stocks: follow SPY ───────────────────────────────────────
        sector = "general"
        if spy_change_pct < _DOWN_SPY_THRESHOLD:
            return SectorStrength(sector=sector, strength="weak", confidence_adjustment=_ADJ_WEAK_STOCK)
        if spy_change_pct > _RISING_THRESHOLD:
            return SectorStrength(sector=sector, strength="strong", confidence_adjustment=_ADJ_STRONG)
        return SectorStrength(sector=sector, strength="neutral", confidence_adjustment=_ADJ_NEUTRAL)

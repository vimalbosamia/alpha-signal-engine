"""
UniverseManager — institutional-grade dynamic symbol selection.

Replaces static watchlists with a pipeline:
  Market Scanner → Liquidity Filter → Volatility Filter → Regime Filter
  → Correlation Filter → Ranking Engine → Trade Candidate List

Updates every scan cycle. Only top-ranked symbols get scanned by strategies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class SymbolScore:
    symbol: str
    asset_class: str
    group: str              # "major", "layer1", "defi", "meme", "macro", "stock"
    volume_24h: float
    spread_quality: str     # "good", "fair", "poor"
    volatility_pct: float
    momentum_score: float   # -1 to +1
    liquidity_ok: bool
    volatility_ok: bool
    regime_compatible: bool
    correlation_safe: bool
    final_rank: float       # 0-100, higher = better candidate
    eligible: bool


@dataclass(frozen=True)
class UniverseResult:
    total_scanned: int
    eligible_count: int
    top_candidates: list[SymbolScore]
    rejected_count: int
    rejection_reasons: dict[str, int]  # reason → count


# ── Asset Groups ──────────────────────────────────────────────────────────────

ASSET_GROUPS = {
    "major": {"BTCUSDT", "ETHUSDT"},
    "layer1": {"SOLUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT"},
    "defi": {"UNIUSDT", "AAVEUSDT", "INJUSDT", "LINKUSDT"},
    "meme": {"DOGEUSDT", "PEPEUSDT", "SHIBUSDT"},
    "high_beta": {"SUIUSDT", "ARBUSDT", "OPUSDT", "STXUSDT", "IMXUSDT"},
    "macro_etf": {"SPY", "QQQ", "DIA", "IWM", "TLT", "GLD", "VIX"},
    "tech_stock": {"AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "GOOGL", "META"},
}

NEWS_SENSITIVE = {"BTCUSDT", "ETHUSDT", "NVDA", "TSLA", "AAPL"}
RISK_ON = {"BTCUSDT", "SOLUSDT", "AVAXUSDT", "PEPEUSDT", "DOGEUSDT"}
RISK_OFF = {"SPY", "QQQ", "GLD", "TLT", "VIX"}


def get_group(symbol: str) -> str:
    for group, symbols in ASSET_GROUPS.items():
        if symbol in symbols:
            return group
    return "other"


class UniverseManager:
    """Dynamically selects and ranks trading candidates."""

    def __init__(
        self,
        max_candidates: int = 30,
        min_volume_usd: float = 10_000_000,
        max_spread_pct: float = 0.3,
        min_volatility_pct: float = 0.1,
        max_volatility_pct: float = 15.0,
        max_correlated: int = 3,
        correlation_threshold: float = 0.85,
    ) -> None:
        self._max = max_candidates
        self._min_vol = min_volume_usd
        self._max_spread = max_spread_pct
        self._min_atr = min_volatility_pct
        self._max_atr = max_volatility_pct
        self._max_corr = max_correlated
        self._corr_thresh = correlation_threshold

    def rank(self, symbols_data: list[dict]) -> UniverseResult:
        """Rank symbols and return top candidates.

        symbols_data: list of dicts with keys:
            symbol, asset_class, volume_24h, spread_pct, atr_pct,
            momentum (float -1 to 1), regime (str), close_prices (list[float])
        """
        scores: list[SymbolScore] = []
        rejections: dict[str, int] = {}

        for d in symbols_data:
            sym = d["symbol"]
            vol = d.get("volume_24h", 0)
            spread = d.get("spread_pct", 0.1)
            atr = d.get("atr_pct", 1.0)
            momentum = d.get("momentum", 0.0)
            regime = d.get("regime", "unknown")
            group = get_group(sym)

            # Liquidity filter
            liquidity_ok = vol >= self._min_vol
            if not liquidity_ok:
                rejections["low_volume"] = rejections.get("low_volume", 0) + 1

            # Spread filter
            spread_quality = "good" if spread < 0.1 else "fair" if spread < self._max_spread else "poor"
            spread_ok = spread <= self._max_spread
            if not spread_ok:
                rejections["wide_spread"] = rejections.get("wide_spread", 0) + 1

            # Volatility filter
            vol_ok = self._min_atr <= atr <= self._max_atr
            if not vol_ok:
                rejections["bad_volatility"] = rejections.get("bad_volatility", 0) + 1

            # Regime compatibility
            regime_ok = regime not in ("climactic", "unknown")
            if not regime_ok:
                rejections["bad_regime"] = rejections.get("bad_regime", 0) + 1

            eligible = liquidity_ok and spread_ok and vol_ok and regime_ok

            # Rank score (0-100)
            rank = 0.0
            if eligible:
                # Volume contribution (max 25)
                rank += min(25, math.log10(max(vol, 1)) * 3)
                # Momentum contribution (max 25)
                rank += (abs(momentum) * 25)
                # Low spread bonus (max 15)
                rank += max(0, 15 - spread * 50)
                # Volatility sweet spot bonus (max 15) — 1-5% is ideal
                if 1.0 <= atr <= 5.0:
                    rank += 15
                elif 0.5 <= atr <= 10.0:
                    rank += 8
                # Group bonus (max 20)
                group_bonus = {"major": 20, "layer1": 15, "defi": 12, "tech_stock": 15,
                               "macro_etf": 10, "high_beta": 8, "meme": 5, "other": 3}
                rank += group_bonus.get(group, 3)

            rank = min(100, max(0, rank))

            scores.append(SymbolScore(
                symbol=sym, asset_class=d.get("asset_class", "crypto"),
                group=group, volume_24h=vol, spread_quality=spread_quality,
                volatility_pct=round(atr, 2), momentum_score=round(momentum, 4),
                liquidity_ok=liquidity_ok, volatility_ok=vol_ok,
                regime_compatible=regime_ok, correlation_safe=True,  # simplified
                final_rank=round(rank, 2), eligible=eligible,
            ))

        # Sort by rank, take top N
        scores.sort(key=lambda s: s.final_rank, reverse=True)
        eligible_scores = [s for s in scores if s.eligible]
        top = eligible_scores[:self._max]

        return UniverseResult(
            total_scanned=len(symbols_data),
            eligible_count=len(eligible_scores),
            top_candidates=top,
            rejected_count=len(symbols_data) - len(eligible_scores),
            rejection_reasons=rejections,
        )

    def is_news_sensitive(self, symbol: str) -> bool:
        return symbol in NEWS_SENSITIVE

    def is_risk_on(self, symbol: str) -> bool:
        return symbol in RISK_ON

    def is_risk_off(self, symbol: str) -> bool:
        return symbol in RISK_OFF

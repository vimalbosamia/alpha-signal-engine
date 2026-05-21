"""
libs.learning.vector_memory — Vector Memory + Similarity Engine.

Stores trade context as fixed-length numeric vectors and retrieves the
K most similar historical conditions before entering new trades.  Uses
numpy-only cosine similarity (no external vector DB required).

Design rules:
  - All public methods are pure / side-effect-free except store mutation
  - No magic numbers — all thresholds are named class constants
  - JSON persistence for portability across sessions
  - Module-level singleton via get_vector_memory()
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from libs.core.logging.logger import get_logger

_log = get_logger(__name__)


# ── Feature definition ───────────────────────────────────────────────────────

VECTOR_FEATURE_NAMES: list[str] = [
    "rsi",
    "macd_histogram",
    "atr_pct",
    "volume_ratio",           # volume / avg_volume
    "regime_id",              # integer encoding of MarketRegime
    "volatility_percentile",  # 0-100
    "ema_alignment",          # fast_ema / slow_ema ratio
    "bb_position",            # (close - bb_lower) / (bb_upper - bb_lower)
    "trend_strength",         # ADX or equivalent 0-100
    "hour_sin",               # cyclical time encoding
    "hour_cos",
    "day_of_week",            # 0-6
    "spread_pct",             # bid-ask spread as % of price
    "funding_rate",           # perp funding rate (0 for spot)
    "open_interest_change",   # OI change % (0 if unavailable)
]

VECTOR_DIM: int = len(VECTOR_FEATURE_NAMES)

# Regime encoding map — covers all 17 regimes
REGIME_ENCODING: dict[str, int] = {
    "trending_up": 0,
    "trending_down": 1,
    "ranging_low_vol": 2,
    "ranging_high_vol": 3,
    "breakout": 4,
    "climactic": 5,
    "accumulation": 6,
    "distribution": 7,
    "panic_selloff": 8,
    "liquidation_event": 9,
    "reversal": 10,
    "low_liquidity": 11,
    "compression": 12,
    "expansion": 13,
    "news_driven": 14,
    "mean_reversion": 15,
    "unknown": 16,
}


# ── Helper: build vector from market context ─────────────────────────────────

def build_context_vector(context: dict[str, Any]) -> np.ndarray:
    """Build a VECTOR_DIM-length numpy array from a market context dict.

    Missing keys default to 0.0.  Regime strings are encoded via
    REGIME_ENCODING.
    """
    regime_str = str(context.get("regime", context.get("market_regime", "unknown")))
    regime_id = float(REGIME_ENCODING.get(regime_str, REGIME_ENCODING["unknown"]))

    hour = float(context.get("hour", 0))
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    vec = np.array([
        float(context.get("rsi", 50.0)),
        float(context.get("macd_histogram", 0.0)),
        float(context.get("atr_pct", 0.0)),
        float(context.get("volume_ratio", 1.0)),
        regime_id,
        float(context.get("volatility_percentile", 50.0)),
        float(context.get("ema_alignment", 1.0)),
        float(context.get("bb_position", 0.5)),
        float(context.get("trend_strength", 0.0)),
        hour_sin,
        hour_cos,
        float(context.get("day_of_week", 0)),
        float(context.get("spread_pct", 0.0)),
        float(context.get("funding_rate", 0.0)),
        float(context.get("open_interest_change", 0.0)),
    ], dtype=np.float64)

    return vec


# ── Stored trade entry ───────────────────────────────────────────────────────

class VectorEntry:
    """A single stored trade vector with its outcome metadata."""

    __slots__ = ("vector", "trade_id", "symbol", "strategy", "won", "pnl", "rr", "regime", "timestamp")

    def __init__(
        self,
        vector: np.ndarray,
        trade_id: str,
        symbol: str = "",
        strategy: str = "",
        won: bool = False,
        pnl: float = 0.0,
        rr: float = 0.0,
        regime: str = "unknown",
        timestamp: str = "",
    ) -> None:
        self.vector = vector
        self.trade_id = trade_id
        self.symbol = symbol
        self.strategy = strategy
        self.won = won
        self.pnl = pnl
        self.rr = rr
        self.regime = regime
        self.timestamp = timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": self.vector.tolist(),
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "won": self.won,
            "pnl": self.pnl,
            "rr": self.rr,
            "regime": self.regime,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VectorEntry":
        return cls(
            vector=np.array(d["vector"], dtype=np.float64),
            trade_id=d.get("trade_id", ""),
            symbol=d.get("symbol", ""),
            strategy=d.get("strategy", ""),
            won=d.get("won", False),
            pnl=d.get("pnl", 0.0),
            rr=d.get("rr", 0.0),
            regime=d.get("regime", "unknown"),
            timestamp=d.get("timestamp", ""),
        )


# ── Similarity result ────────────────────────────────────────────────────────

class SimilarityResult:
    """Result of a KNN similarity query."""

    __slots__ = ("entry", "similarity")

    def __init__(self, entry: VectorEntry, similarity: float) -> None:
        self.entry = entry
        self.similarity = similarity

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.entry.trade_id,
            "symbol": self.entry.symbol,
            "strategy": self.entry.strategy,
            "won": self.entry.won,
            "pnl": self.entry.pnl,
            "rr": self.entry.rr,
            "regime": self.entry.regime,
            "similarity": round(self.similarity, 4),
        }


# ── VectorMemory ─────────────────────────────────────────────────────────────

class VectorMemory:
    """Vector-based trade memory with cosine similarity search.

    Parameters
    ----------
    max_entries: Maximum stored vectors before oldest are evicted.
    """

    MAX_ENTRIES_DEFAULT: int = 10_000
    DEFAULT_K: int = 5

    def __init__(self, max_entries: int = MAX_ENTRIES_DEFAULT) -> None:
        self._entries: list[VectorEntry] = []
        self._max_entries = max_entries

    @property
    def size(self) -> int:
        return len(self._entries)

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(
        self,
        context: dict[str, Any],
        trade_id: str,
        symbol: str = "",
        strategy: str = "",
        won: bool = False,
        pnl: float = 0.0,
        rr: float = 0.0,
        regime: str = "unknown",
        timestamp: str = "",
    ) -> None:
        """Store a trade's market context vector.

        Parameters
        ----------
        context: Dict of market features (see VECTOR_FEATURE_NAMES).
        trade_id: Unique identifier for the trade.
        symbol: Trading symbol.
        strategy: Strategy name.
        won: Trade outcome.
        pnl: Realised P&L.
        rr: Risk-reward ratio.
        regime: Market regime at entry.
        timestamp: ISO timestamp string.
        """
        vec = build_context_vector(context)
        entry = VectorEntry(
            vector=vec,
            trade_id=trade_id,
            symbol=symbol,
            strategy=strategy,
            won=won,
            pnl=pnl,
            rr=rr,
            regime=regime,
            timestamp=timestamp,
        )
        self._entries.append(entry)

        # Evict oldest if over limit
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

        _log.debug(
            "vector_memory.stored",
            trade_id=trade_id,
            total_entries=len(self._entries),
        )

    def store_raw(self, entry: VectorEntry) -> None:
        """Store a pre-built VectorEntry directly."""
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries:]

    # ── Query ─────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        context: dict[str, Any],
        k: int | None = None,
        min_similarity: float = 0.0,
        filter_symbol: str | None = None,
        filter_strategy: str | None = None,
    ) -> list[SimilarityResult]:
        """Find K most similar historical trade conditions.

        Parameters
        ----------
        context: Current market context dict.
        k: Number of neighbors to return (default DEFAULT_K).
        min_similarity: Minimum cosine similarity threshold.
        filter_symbol: Only match this symbol (None = all).
        filter_strategy: Only match this strategy (None = all).

        Returns
        -------
        List of SimilarityResult sorted by descending similarity.
        """
        if not self._entries:
            return []

        k = k or self.DEFAULT_K
        query_vec = build_context_vector(context)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0.0:
            return []

        results: list[SimilarityResult] = []

        for entry in self._entries:
            if filter_symbol and entry.symbol != filter_symbol:
                continue
            if filter_strategy and entry.strategy != filter_strategy:
                continue

            entry_norm = np.linalg.norm(entry.vector)
            if entry_norm == 0.0:
                continue

            similarity = float(
                np.dot(query_vec, entry.vector) / (query_norm * entry_norm)
            )

            if similarity >= min_similarity:
                results.append(SimilarityResult(entry=entry, similarity=similarity))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:k]

    def find_similar_batch(
        self,
        context: dict[str, Any],
        k: int | None = None,
        min_similarity: float = 0.0,
    ) -> list[SimilarityResult]:
        """Vectorized batch similarity search (faster for large stores).

        Uses matrix multiplication for all-at-once cosine similarity.
        """
        if not self._entries:
            return []

        k = k or self.DEFAULT_K
        query_vec = build_context_vector(context)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0.0:
            return []

        # Build matrix of all stored vectors
        matrix = np.vstack([e.vector for e in self._entries])
        norms = np.linalg.norm(matrix, axis=1)

        # Avoid division by zero
        valid_mask = norms > 0.0
        similarities = np.zeros(len(self._entries))
        similarities[valid_mask] = (
            matrix[valid_mask] @ query_vec / (norms[valid_mask] * query_norm)
        )

        # Get top-k indices
        if min_similarity > 0.0:
            valid_mask &= similarities >= min_similarity

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_sims = similarities[valid_indices]
        top_k_idx = valid_indices[np.argsort(valid_sims)[::-1][:k]]

        return [
            SimilarityResult(
                entry=self._entries[int(i)],
                similarity=float(similarities[i]),
            )
            for i in top_k_idx
        ]

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_win_rate_for_similar(
        self,
        context: dict[str, Any],
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> dict[str, Any]:
        """Estimate win probability from similar historical conditions.

        Returns
        -------
        Dict with keys: win_rate, avg_pnl, avg_rr, sample_size,
        avg_similarity, historical_edge.
        """
        similar = self.find_similar_batch(context, k=k, min_similarity=min_similarity)

        if not similar:
            return {
                "win_rate": 0.5,
                "avg_pnl": 0.0,
                "avg_rr": 0.0,
                "sample_size": 0,
                "avg_similarity": 0.0,
                "historical_edge": 0.0,
            }

        wins = sum(1 for r in similar if r.entry.won)
        total = len(similar)
        avg_pnl = sum(r.entry.pnl for r in similar) / total
        avg_rr = sum(r.entry.rr for r in similar) / total
        avg_sim = sum(r.similarity for r in similar) / total
        win_rate = wins / total

        # Historical edge = weighted win_rate by similarity
        weighted_wins = sum(r.similarity for r in similar if r.entry.won)
        weighted_total = sum(r.similarity for r in similar)
        historical_edge = weighted_wins / weighted_total if weighted_total > 0 else 0.5

        return {
            "win_rate": round(win_rate, 4),
            "avg_pnl": round(avg_pnl, 4),
            "avg_rr": round(avg_rr, 4),
            "sample_size": total,
            "avg_similarity": round(avg_sim, 4),
            "historical_edge": round(historical_edge, 4),
        }

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics of the vector memory store."""
        if not self._entries:
            return {
                "total_entries": 0,
                "unique_symbols": 0,
                "unique_strategies": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
            }

        symbols = set(e.symbol for e in self._entries)
        strategies = set(e.strategy for e in self._entries)
        wins = sum(1 for e in self._entries if e.won)
        total_pnl = sum(e.pnl for e in self._entries)

        return {
            "total_entries": len(self._entries),
            "unique_symbols": len(symbols),
            "unique_strategies": len(strategies),
            "win_rate": round(wins / len(self._entries), 4),
            "avg_pnl": round(total_pnl / len(self._entries), 4),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist all vectors to a JSON file.

        Parameters
        ----------
        path: Destination file path (parent directories are created).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "max_entries": self._max_entries,
            "entries": [e.to_dict() for e in self._entries],
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _log.info("vector_memory.saved", path=str(dest), entries=len(self._entries))

    def load(self, path: str | Path) -> None:
        """Restore vectors from a previously saved JSON file.

        Parameters
        ----------
        path: Source file path.
        """
        src = Path(path)
        if not src.exists():
            _log.debug("vector_memory.no_file", path=str(src))
            return

        payload: dict[str, Any] = json.loads(src.read_text(encoding="utf-8"))
        self._max_entries = payload.get("max_entries", self._max_entries)
        self._entries = [
            VectorEntry.from_dict(d) for d in payload.get("entries", [])
        ]
        _log.info(
            "vector_memory.loaded",
            path=str(src),
            entries=len(self._entries),
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_singleton: VectorMemory | None = None


def get_vector_memory() -> VectorMemory:
    """Return the module-level singleton VectorMemory.

    The instance is created lazily on the first call.
    """
    global _singleton
    if _singleton is None:
        _singleton = VectorMemory()
        _log.info("vector_memory.singleton_initialised")
    return _singleton

"""
libs.learning.pattern_scorer — Pattern+regime win-rate tracking with decay.

Tracks win/loss statistics per (pattern_name, regime) combo.  Every closed
trade reports its patterns and regime.  The store accumulates statistics,
applies exponential decay so stale data loses influence, and provides
confidence adjustments for future trades.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from libs.core.logging.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_TRADES_FOR_SCORE: int = 5
BASELINE_WIN_RATE: float = 0.45
MAX_CONFIDENCE_ADJ: float = 0.15
DECAY_INTERVAL_TRADES: int = 50


# ── Data Class ────────────────────────────────────────────────────────────────

@dataclass
class PatternScore:
    """Accumulated statistics for one (pattern_name, regime) combination.

    wins, losses, total_rr, and effective_weight are all floats because
    they are multiplied by the decay alpha on each decay pass.
    """

    pattern_name: str
    regime: str
    wins: float = 0.0
    losses: float = 0.0
    total_rr: float = 0.0
    effective_weight: float = 0.0
    last_updated: str = ""

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        """Rounded integer count of recorded trades."""
        return round(self.wins + self.losses)

    @property
    def win_rate(self) -> float:
        """Fraction of wins; 0.0 when no trades recorded."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def avg_rr(self) -> float:
        """Average risk-reward ratio; 0.0 when no trades recorded."""
        total = self.wins + self.losses
        return self.total_rr / total if total > 0 else 0.0


# ── Store ─────────────────────────────────────────────────────────────────────

class PatternScoreStore:
    """Tracks win/loss statistics per (pattern_name, regime) combo.

    Keys in the internal dict are ``"pattern_name|regime"`` strings.
    Every *DECAY_INTERVAL_TRADES* recorded outcomes the store automatically
    multiplies all accumulators by *decay_alpha* so that older data gradually
    loses influence relative to recent results.
    """

    def __init__(self, decay_alpha: float = 0.97) -> None:
        self._alpha: float = decay_alpha
        self._scores: dict[str, PatternScore] = {}
        self._trade_count: int = 0  # global counter used for auto-decay trigger

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _key(pattern: str, regime: str) -> str:
        return f"{pattern}|{regime}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(self, pattern: str, regime: str, won: bool, rr: float) -> None:
        """Record a single trade outcome for a (pattern, regime) pair.

        Auto-decay is triggered every *DECAY_INTERVAL_TRADES* calls.

        Parameters
        ----------
        pattern: Pattern name (e.g. "hammer", "engulfing_bull").
        regime:  Market regime label (e.g. "trending_up", "ranging").
        won:     True if the trade was a win.
        rr:      Realised risk-reward ratio for the trade.
        """
        key = self._key(pattern, regime)
        if key not in self._scores:
            self._scores[key] = PatternScore(pattern_name=pattern, regime=regime)

        score = self._scores[key]
        if won:
            score.wins += 1.0
        else:
            score.losses += 1.0
        score.total_rr += rr
        score.effective_weight += 1.0
        score.last_updated = self._utc_now()

        self._trade_count += 1
        if self._trade_count % DECAY_INTERVAL_TRADES == 0:
            log.debug(
                "pattern_scorer.auto_decay",
                trade_count=self._trade_count,
                alpha=self._alpha,
            )
            self.apply_decay()

    def get_score(self, pattern: str, regime: str) -> PatternScore:
        """Return the PatternScore for (pattern, regime), or an empty one."""
        key = self._key(pattern, regime)
        return self._scores.get(
            key,
            PatternScore(pattern_name=pattern, regime=regime),
        )

    def apply_decay(self) -> None:
        """Multiply wins, losses, total_rr, and effective_weight by alpha.

        This ages out stale data so recent results have proportionally
        greater influence on confidence calculations.
        """
        for score in self._scores.values():
            score.wins *= self._alpha
            score.losses *= self._alpha
            score.total_rr *= self._alpha
            score.effective_weight *= self._alpha

    def confidence_adjustment(self, pattern: str, regime: str) -> float:
        """Return a confidence adjustment in [-MAX_CONFIDENCE_ADJ, +MAX_CONFIDENCE_ADJ].

        Returns 0.0 when the pattern has fewer than MIN_TRADES_FOR_SCORE trades.
        Positive when win_rate > BASELINE_WIN_RATE; negative otherwise.
        Scaled linearly from 0 → ±MAX_CONFIDENCE_ADJ across [0, 1] win-rate range.
        """
        score = self.get_score(pattern, regime)
        if score.total_trades < MIN_TRADES_FOR_SCORE:
            return 0.0

        deviation = score.win_rate - BASELINE_WIN_RATE
        # Normalise: max possible deviation is 1 - BASELINE (positive) or
        # 0 - BASELINE (negative).  Scale uniformly so the full ±MAX is
        # reachable at the extremes.
        if deviation >= 0:
            max_deviation = 1.0 - BASELINE_WIN_RATE
        else:
            max_deviation = BASELINE_WIN_RATE

        raw_adj = (deviation / max_deviation) * MAX_CONFIDENCE_ADJ
        # Clamp to [-MAX_CONFIDENCE_ADJ, +MAX_CONFIDENCE_ADJ] for safety.
        return max(-MAX_CONFIDENCE_ADJ, min(MAX_CONFIDENCE_ADJ, raw_adj))

    def get_top_patterns(
        self,
        regime: str,
        n: int = 5,
    ) -> list[PatternScore]:
        """Return the top *n* patterns for *regime* sorted by win_rate descending.

        Only patterns with at least MIN_TRADES_FOR_SCORE trades are included.
        """
        candidates = self._regime_candidates(regime)
        return sorted(candidates, key=lambda s: s.win_rate, reverse=True)[:n]

    def get_worst_patterns(
        self,
        regime: str,
        n: int = 5,
    ) -> list[PatternScore]:
        """Return the worst *n* patterns for *regime* sorted by win_rate ascending.

        Only patterns with at least MIN_TRADES_FOR_SCORE trades are included.
        """
        candidates = self._regime_candidates(regime)
        return sorted(candidates, key=lambda s: s.win_rate)[:n]

    def _regime_candidates(self, regime: str) -> list[PatternScore]:
        """Return all scores for *regime* that meet the minimum trade threshold."""
        return [
            s
            for s in self._scores.values()
            if s.regime == regime and s.total_trades >= MIN_TRADES_FOR_SCORE
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the store to a JSON file at *path*.

        Creates parent directories if they do not exist.
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "decay_alpha": self._alpha,
            "trade_count": self._trade_count,
            "scores": {key: asdict(score) for key, score in self._scores.items()},
        }
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.debug("pattern_scorer.saved", path=str(dest), entries=len(self._scores))

    @classmethod
    def load(cls, path: str | Path) -> "PatternScoreStore":
        """Load a previously saved store from *path*.

        Returns an empty store if the file does not exist.
        """
        src = Path(path)
        if not src.exists():
            log.warning("pattern_scorer.load_missing", path=str(src))
            return cls()

        payload = json.loads(src.read_text(encoding="utf-8"))
        store = cls(decay_alpha=payload.get("decay_alpha", 0.97))
        store._trade_count = payload.get("trade_count", 0)
        for key, raw in payload.get("scores", {}).items():
            store._scores[key] = PatternScore(**raw)
        log.debug(
            "pattern_scorer.loaded",
            path=str(src),
            entries=len(store._scores),
        )
        return store

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def get_all_stats(self) -> dict:
        """Return a summary dict suitable for dashboard display.

        Structure
        ---------
        {
            "total_patterns": int,
            "total_trades_recorded": int,
            "by_regime": {
                "<regime>": [
                    {
                        "pattern": str,
                        "total_trades": int,
                        "win_rate": float,
                        "avg_rr": float,
                        "effective_weight": float,
                    }, ...
                ]
            }
        }
        """
        by_regime: dict[str, list[dict]] = {}
        for score in self._scores.values():
            regime = score.regime
            if regime not in by_regime:
                by_regime[regime] = []
            by_regime[regime].append(
                {
                    "pattern": score.pattern_name,
                    "total_trades": score.total_trades,
                    "win_rate": round(score.win_rate, 4),
                    "avg_rr": round(score.avg_rr, 4),
                    "effective_weight": round(score.effective_weight, 4),
                }
            )

        return {
            "total_patterns": len(self._scores),
            "total_trades_recorded": self._trade_count,
            "by_regime": by_regime,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_store: PatternScoreStore | None = None


def get_pattern_store() -> PatternScoreStore:
    """Return the module-level singleton PatternScoreStore.

    The instance is created lazily on the first call.
    """
    global _store
    if _store is None:
        _store = PatternScoreStore()
    return _store

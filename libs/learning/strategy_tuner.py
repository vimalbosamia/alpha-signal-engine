"""
libs.learning.strategy_tuner — Auto-adjusts strategy thresholds based on trade outcomes.

Analyzes closed trade history per strategy and tunes min_confidence and
min_rr thresholds to improve signal quality over time.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_FLOOR: float = 0.45
CONFIDENCE_CEILING: float = 0.85
CONFIDENCE_DEFAULT: float = 0.60

RR_FLOOR: float = 1.0
RR_CEILING: float = 3.0
RR_DEFAULT: float = 1.5

MIN_TRADES_TO_TUNE: int = 8
TUNE_STEP: float = 0.03
RR_TUNE_STEP: float = 0.15

_MAX_HISTORY_PER_STRATEGY: int = 200


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class StrategyParams:
    """Mutable tunable thresholds for a single strategy."""

    min_confidence: float = CONFIDENCE_DEFAULT
    min_rr: float = RR_DEFAULT
    tune_count: int = 0


@dataclass
class TradeRecord:
    """Immutable record of one closed trade outcome."""

    won: bool
    confidence: float
    rr: float


# ── Tuner ─────────────────────────────────────────────────────────────────────

class StrategyParameterTuner:
    """Auto-adjusts per-strategy confidence and R:R thresholds from trade history.

    State is held in two plain dicts:
    - ``_params``  — strategy name → :class:`StrategyParams`
    - ``_history`` — strategy name → list of :class:`TradeRecord`
    """

    def __init__(self) -> None:
        self._params: dict[str, StrategyParams] = {}
        self._history: defaultdict[str, list[TradeRecord]] = defaultdict(list)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_params(self, strategy: str) -> StrategyParams:
        """Return existing params for *strategy*, creating defaults if absent."""
        if strategy not in self._params:
            self._params[strategy] = StrategyParams()
        return self._params[strategy]

    def record_outcome(
        self,
        strategy: str,
        won: bool,
        confidence: float,
        rr: float,
    ) -> None:
        """Append one closed trade outcome to history for *strategy*."""
        self._history[strategy].append(TradeRecord(won=won, confidence=confidence, rr=rr))

    def tune(self, strategy: str) -> StrategyParams:
        """Analyze history for *strategy* and adjust its thresholds.

        Rules
        -----
        Confidence:
        - win_rate < 0.40                      → raise min_confidence by TUNE_STEP,
                                                 cap at CONFIDENCE_CEILING
        - win_rate > 0.60 and 12+ trades       → lower min_confidence by TUNE_STEP*0.5,
                                                 floor at CONFIDENCE_FLOOR

        R:R:
        - avg_loss_rr < 1.5 and win_rate < 0.50 → raise min_rr by RR_TUNE_STEP,
                                                   cap at RR_CEILING
        - avg_win_rr > 2.0 and win_rate > 0.55  → lower min_rr by RR_TUNE_STEP*0.5,
                                                   floor at RR_FLOOR

        Requires at least MIN_TRADES_TO_TUNE (8) trades; returns params unchanged
        if the history is too short.  Increments tune_count on every qualifying call.
        """
        try:
            from libs.core.logging.logger import get_logger
            log = get_logger(__name__)
        except Exception:
            log = None  # type: ignore[assignment]

        history = self._history[strategy]
        params = self.get_params(strategy)

        if len(history) < MIN_TRADES_TO_TUNE:
            if log:
                log.debug(
                    "strategy_tuner.skip",
                    strategy=strategy,
                    trades=len(history),
                    required=MIN_TRADES_TO_TUNE,
                )
            return params

        total = len(history)
        wins = [r for r in history if r.won]
        losses = [r for r in history if not r.won]
        win_rate = len(wins) / total

        # ── Confidence tuning ────────────────────────────────────────────────
        if win_rate < 0.40:
            params.min_confidence = min(
                params.min_confidence + TUNE_STEP,
                CONFIDENCE_CEILING,
            )
        elif win_rate > 0.60 and total >= 12:
            params.min_confidence = max(
                params.min_confidence - TUNE_STEP * 0.5,
                CONFIDENCE_FLOOR,
            )

        # ── R:R tuning ───────────────────────────────────────────────────────
        avg_loss_rr = (
            sum(r.rr for r in losses) / len(losses) if losses else float("inf")
        )
        avg_win_rr = (
            sum(r.rr for r in wins) / len(wins) if wins else 0.0
        )

        if avg_loss_rr < 1.5 and win_rate < 0.50:
            params.min_rr = min(params.min_rr + RR_TUNE_STEP, RR_CEILING)
        elif avg_win_rr > 2.0 and win_rate > 0.55:
            params.min_rr = max(params.min_rr - RR_TUNE_STEP * 0.5, RR_FLOOR)

        params.tune_count += 1

        if log:
            log.info(
                "strategy_tuner.tuned",
                strategy=strategy,
                win_rate=round(win_rate, 3),
                min_confidence=round(params.min_confidence, 4),
                min_rr=round(params.min_rr, 4),
                tune_count=params.tune_count,
            )

        return params

    def tune_all(self) -> dict[str, StrategyParams]:
        """Tune every strategy that has enough history.

        Returns a mapping of strategy name → updated :class:`StrategyParams`.
        """
        results: dict[str, StrategyParams] = {}
        for strategy, history in self._history.items():
            if len(history) >= MIN_TRADES_TO_TUNE:
                results[strategy] = self.tune(strategy)
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize tuner state to a JSON file at *path*.

        Only the last ``_MAX_HISTORY_PER_STRATEGY`` records are retained per
        strategy to keep the file size bounded.
        """
        data: dict[str, Any] = {
            "params": {
                name: asdict(p) for name, p in self._params.items()
            },
            "history": {
                name: [asdict(r) for r in records[-_MAX_HISTORY_PER_STRATEGY:]]
                for name, records in self._history.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    def load(self, path: str | Path) -> None:
        """Load tuner state from a JSON file previously written by :meth:`save`.

        Replaces all current in-memory state.
        """
        path = Path(path)
        data: dict[str, Any] = json.loads(path.read_text())

        self._params = {
            name: StrategyParams(**p)
            for name, p in data.get("params", {}).items()
        }
        self._history = defaultdict(list)
        for name, records in data.get("history", {}).items():
            self._history[name] = [TradeRecord(**r) for r in records]

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Return a summary dict suitable for dashboard display.

        Each key is a strategy name; the value includes current params and
        aggregate statistics derived from the trade history.
        """
        stats: dict[str, dict[str, Any]] = {}
        all_strategies = set(self._params) | set(self._history)

        for strategy in sorted(all_strategies):
            params = self.get_params(strategy)
            history = self._history[strategy]
            total = len(history)
            wins = sum(1 for r in history if r.won)
            win_rate = wins / total if total > 0 else 0.0
            avg_rr = sum(r.rr for r in history) / total if total > 0 else 0.0

            stats[strategy] = {
                "min_confidence": params.min_confidence,
                "min_rr": params.min_rr,
                "tune_count": params.tune_count,
                "total_trades": total,
                "win_rate": round(win_rate, 4),
                "avg_rr": round(avg_rr, 4),
            }

        return stats


# ── Module-level singleton ────────────────────────────────────────────────────

_tuner_instance: StrategyParameterTuner | None = None


def get_strategy_tuner() -> StrategyParameterTuner:
    """Return the process-wide :class:`StrategyParameterTuner` singleton."""
    global _tuner_instance
    if _tuner_instance is None:
        _tuner_instance = StrategyParameterTuner()
    return _tuner_instance

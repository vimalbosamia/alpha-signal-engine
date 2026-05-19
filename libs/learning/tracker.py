"""
libs.learning.tracker — Performance tracking and weight calibration.

Tracks win/loss per strategy, indicator, and pattern.  Provides weight
update suggestions and a safety gate that blocks dangerous changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class PerformanceRecord:
    """Mutable record of win/loss statistics for one tracked entity."""

    name: str
    category: str          # "strategy", "indicator", "pattern"
    total: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    last_updated: str = ""

    # Running sum kept private so avg_pnl is always consistent.
    _total_pnl: float = field(default=0.0, repr=False, compare=False)

    def _refresh(self, timestamp: str) -> None:
        """Recompute derived fields in-place after a new outcome is recorded."""
        self.win_rate = self.wins / self.total if self.total > 0 else 0.0
        self.avg_pnl = self._total_pnl / self.total if self.total > 0 else 0.0
        self.last_updated = timestamp


@dataclass(frozen=True)
class WeightUpdate:
    """Immutable suggestion to adjust the weight of one entity."""

    name: str
    category: str
    old_weight: float
    new_weight: float
    reason: str


@dataclass(frozen=True)
class SafetyGateResult:
    """Immutable outcome of the safety gate check."""

    approved: bool
    reason: str
    checks_passed: list[str]
    checks_failed: list[str]


# ── Tracker ───────────────────────────────────────────────────────────────────

class PerformanceTracker:
    """Tracks win/loss per strategy, indicator, and pattern.

    All internal state is held in a plain dict keyed by ``(name, category)``.
    No I/O, no external dependencies.
    """

    # Maximum allowed single-step weight change (30 %).
    _MAX_WEIGHT_DELTA: float = 0.30
    # Win-rate threshold above which we increase the weight.
    _HIGH_WIN_RATE: float = 0.60
    # Win-rate threshold below which we decrease the weight.
    _LOW_WIN_RATE: float = 0.35
    # Weight increase factor (10 %).
    _INCREASE_FACTOR: float = 0.10
    # Weight decrease factor (20 %).
    _DECREASE_FACTOR: float = 0.20
    # Acceptable deviation of weight sum from 1.0 (5 %).
    _SUM_TOLERANCE: float = 0.05

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], PerformanceRecord] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def record_outcome(
        self,
        name: str,
        category: str,
        won: bool,
        pnl: float = 0.0,
    ) -> None:
        """Record a win or loss for any tracked entity.

        Parameters
        ----------
        name:     Unique identifier (e.g. "RSI", "breakout_long").
        category: One of "strategy", "indicator", "pattern".
        won:      True for a win, False for a loss.
        pnl:      Realised profit/loss for this trade (may be negative).
        """
        key = (name, category)
        if key not in self._records:
            self._records[key] = PerformanceRecord(name=name, category=category)

        rec = self._records[key]
        rec.total += 1
        if won:
            rec.wins += 1
        else:
            rec.losses += 1
        rec._total_pnl += pnl
        rec._refresh(_utc_now())

    def get_record(self, name: str, category: str) -> PerformanceRecord | None:
        """Return the performance record for one entity, or *None* if unseen."""
        return self._records.get((name, category))

    def get_all(self, category: str | None = None) -> list[PerformanceRecord]:
        """Return all records, optionally filtered to a single category.

        Parameters
        ----------
        category: When provided, only records whose ``category`` field matches
                  are returned.  Pass *None* (or omit) to get everything.
        """
        records = list(self._records.values())
        if category is not None:
            records = [r for r in records if r.category == category]
        return records

    def suggest_weight_updates(
        self,
        current_weights: dict[str, float],
        min_trades: int = 20,
    ) -> list[WeightUpdate]:
        """Suggest weight adjustments based on measured performance.

        Rules
        -----
        - Fewer than *min_trades* recorded → no suggestion for that entity.
        - Win-rate > 60 % → increase weight by 10 %.
        - Win-rate < 35 % → decrease weight by 20 %.
        - Proposed delta clamped so the change never exceeds ±30 % of the
          *current* weight.

        Parameters
        ----------
        current_weights: Mapping of entity name → current weight.
        min_trades:      Minimum number of recorded outcomes required before
                         a suggestion is emitted.

        Returns
        -------
        List of :class:`WeightUpdate` suggestions (may be empty).
        """
        updates: list[WeightUpdate] = []

        for name, old_weight in current_weights.items():
            # Collect records across all categories for this name.
            matching = [
                rec
                for rec in self._records.values()
                if rec.name == name
            ]
            if not matching:
                continue

            # Aggregate across categories if multiple records share a name.
            total_trades = sum(r.total for r in matching)
            total_wins = sum(r.wins for r in matching)

            if total_trades < min_trades:
                continue

            win_rate = total_wins / total_trades

            if win_rate > self._HIGH_WIN_RATE:
                raw_delta = old_weight * self._INCREASE_FACTOR
                reason = (
                    f"win_rate={win_rate:.1%} > {self._HIGH_WIN_RATE:.0%}; "
                    "increase weight by 10%"
                )
            elif win_rate < self._LOW_WIN_RATE:
                raw_delta = -old_weight * self._DECREASE_FACTOR
                reason = (
                    f"win_rate={win_rate:.1%} < {self._LOW_WIN_RATE:.0%}; "
                    "decrease weight by 20%"
                )
            else:
                # Win-rate is in the neutral band — no change.
                continue

            # Clamp so the change is never larger than ±30 % of the current weight.
            max_delta = old_weight * self._MAX_WEIGHT_DELTA
            clamped_delta = max(min(raw_delta, max_delta), -max_delta)

            new_weight = old_weight + clamped_delta
            updates.append(
                WeightUpdate(
                    name=name,
                    category=matching[0].category,
                    old_weight=old_weight,
                    new_weight=new_weight,
                    reason=reason,
                )
            )

        return updates

    def safety_gate(
        self,
        new_weights: dict[str, float],
        old_weights: dict[str, float],
    ) -> SafetyGateResult:
        """Validate that proposed weight changes are safe before applying them.

        Checks
        ------
        1. No individual weight changed by more than 30 %.
        2. New weights sum to approximately 1.0 (within 5 %).
        3. No weight is negative.
        4. At least 3 entities in the tracker have ≥ ``min_trades`` data.

        Parameters
        ----------
        new_weights: Proposed weight mapping.
        old_weights: Current weight mapping (baseline for delta checks).

        Returns
        -------
        A :class:`SafetyGateResult` with ``approved=True`` only when all
        checks pass.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []

        # Check 1 — no single weight changed by more than 30 %.
        large_changes = [
            name
            for name, new_w in new_weights.items()
            if name in old_weights
            and abs(new_w - old_weights[name]) > self._MAX_WEIGHT_DELTA * old_weights[name]
        ]
        if large_changes:
            checks_failed.append(
                f"weight change > 30% for: {', '.join(sorted(large_changes))}"
            )
        else:
            checks_passed.append("no weight changed by more than 30%")

        # Check 2 — weights sum to ~1.0.
        weight_sum = sum(new_weights.values())
        if abs(weight_sum - 1.0) > self._SUM_TOLERANCE:
            checks_failed.append(
                f"weights sum to {weight_sum:.4f}, expected 1.0 ± {self._SUM_TOLERANCE}"
            )
        else:
            checks_passed.append(f"weights sum to {weight_sum:.4f} (within tolerance)")

        # Check 3 — no negative weights.
        negative = [n for n, w in new_weights.items() if w < 0]
        if negative:
            checks_failed.append(
                f"negative weights for: {', '.join(sorted(negative))}"
            )
        else:
            checks_passed.append("no negative weights")

        # Check 4 — at least 3 entities have meaningful data.
        entities_with_data = sum(
            1 for rec in self._records.values() if rec.total >= 20
        )
        if entities_with_data < 3:
            checks_failed.append(
                f"only {entities_with_data} entit(ies) have ≥ 20 trades "
                "(need at least 3)"
            )
        else:
            checks_passed.append(
                f"{entities_with_data} entities have sufficient trade data"
            )

        approved = len(checks_failed) == 0
        reason = (
            "all safety checks passed"
            if approved
            else f"blocked: {checks_failed[0]}"
        )

        return SafetyGateResult(
            approved=approved,
            reason=reason,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()

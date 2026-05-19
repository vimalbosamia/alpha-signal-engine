"""WalkForwardValidator — validates strategy performance via rolling train/test splits."""
from __future__ import annotations

from dataclasses import dataclass, field

MIN_TRADES_PER_WINDOW = 10
MIN_WINDOWS = 2
MAX_DEGRADATION = 0.15  # 15% drop train→test is considered overfitting


@dataclass(frozen=True)
class ValidationWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_win_rate: float
    test_win_rate: float
    degradation: float  # train_wr - test_wr (positive = overfitting)


@dataclass(frozen=True)
class ValidationResult:
    strategy_name: str
    total_trades: int
    is_valid: bool
    validation_status: str  # "insufficient_data", "overfitting", "degrading", "validated"
    windows: list[ValidationWindow] = field(default_factory=list)
    avg_train_win_rate: float = 0.0
    avg_test_win_rate: float = 0.0
    avg_degradation: float = 0.0
    confidence_cap: float = 0.50
    explanation: str = ""


def _win_rate(outcomes: list[bool]) -> float:
    if not outcomes:
        return 0.0
    return sum(outcomes) / len(outcomes)


class WalkForwardValidator:
    """Validates strategy performance using rolling train/test windows with overfitting detection."""

    def validate(
        self,
        strategy_name: str,
        outcomes: list[bool],
        train_ratio: float = 0.7,
    ) -> ValidationResult:
        """
        Validate a strategy's outcomes chronologically via walk-forward analysis.

        outcomes: list of True (win) / False (loss) in chronological order.

        Returns a ValidationResult capturing window-level metrics, average
        degradation, and a suggested confidence_cap.
        """
        total = len(outcomes)

        if total < MIN_TRADES_PER_WINDOW * 2:
            return ValidationResult(
                strategy_name=strategy_name,
                total_trades=total,
                is_valid=False,
                validation_status="insufficient_data",
                windows=[],
                avg_train_win_rate=0.0,
                avg_test_win_rate=0.0,
                avg_degradation=0.0,
                confidence_cap=0.50,
                explanation=(
                    f"Only {total} trades — need at least "
                    f"{MIN_TRADES_PER_WINDOW * 2} to validate."
                ),
            )

        windows = self._build_windows(outcomes, train_ratio)

        if len(windows) < MIN_WINDOWS:
            return ValidationResult(
                strategy_name=strategy_name,
                total_trades=total,
                is_valid=False,
                validation_status="insufficient_data",
                windows=windows,
                avg_train_win_rate=0.0,
                avg_test_win_rate=0.0,
                avg_degradation=0.0,
                confidence_cap=0.50,
                explanation=(
                    f"Only {len(windows)} validation window(s) produced — "
                    f"need at least {MIN_WINDOWS}."
                ),
            )

        avg_train_wr = sum(w.train_win_rate for w in windows) / len(windows)
        avg_test_wr = sum(w.test_win_rate for w in windows) / len(windows)
        avg_degradation = sum(w.degradation for w in windows) / len(windows)

        if avg_degradation > MAX_DEGRADATION:
            return ValidationResult(
                strategy_name=strategy_name,
                total_trades=total,
                is_valid=False,
                validation_status="overfitting",
                windows=windows,
                avg_train_win_rate=round(avg_train_wr, 4),
                avg_test_win_rate=round(avg_test_wr, 4),
                avg_degradation=round(avg_degradation, 4),
                confidence_cap=0.55,
                explanation=(
                    f"Avg degradation {avg_degradation:.1%} exceeds threshold "
                    f"{MAX_DEGRADATION:.1%} — strategy likely overfit to training data."
                ),
            )

        if avg_test_wr < 0.40:
            return ValidationResult(
                strategy_name=strategy_name,
                total_trades=total,
                is_valid=False,
                validation_status="degrading",
                windows=windows,
                avg_train_win_rate=round(avg_train_wr, 4),
                avg_test_win_rate=round(avg_test_wr, 4),
                avg_degradation=round(avg_degradation, 4),
                confidence_cap=0.45,
                explanation=(
                    f"Avg test win rate {avg_test_wr:.1%} is below 40% — "
                    "strategy performance is too weak to trust."
                ),
            )

        return ValidationResult(
            strategy_name=strategy_name,
            total_trades=total,
            is_valid=True,
            validation_status="validated",
            windows=windows,
            avg_train_win_rate=round(avg_train_wr, 4),
            avg_test_win_rate=round(avg_test_wr, 4),
            avg_degradation=round(avg_degradation, 4),
            confidence_cap=0.90,
            explanation=(
                f"Strategy validated across {len(windows)} windows. "
                f"Avg test WR: {avg_test_wr:.1%}, avg degradation: {avg_degradation:.1%}."
            ),
        )

    def _build_windows(
        self,
        outcomes: list[bool],
        train_ratio: float,
    ) -> list[ValidationWindow]:
        """
        Build rolling walk-forward windows with 50% step overlap.

        Window size is chosen so both train and test slices contain at least
        MIN_TRADES_PER_WINDOW samples. The window slides forward by half the
        window size each step.
        """
        total = len(outcomes)
        # Minimum window size: enough trades for both train and test portions.
        min_window = int(MIN_TRADES_PER_WINDOW / (1 - train_ratio))
        window_size = max(min_window, MIN_TRADES_PER_WINDOW * 2)
        step = max(1, window_size // 2)

        windows: list[ValidationWindow] = []

        start = 0
        while start + window_size <= total:
            train_end = start + int(window_size * train_ratio)
            test_end = start + window_size

            train_slice = outcomes[start:train_end]
            test_slice = outcomes[train_end:test_end]

            if len(train_slice) < MIN_TRADES_PER_WINDOW or len(test_slice) < MIN_TRADES_PER_WINDOW:
                start += step
                continue

            train_wr = _win_rate(train_slice)
            test_wr = _win_rate(test_slice)
            degradation = train_wr - test_wr

            windows.append(
                ValidationWindow(
                    train_start=start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                    train_win_rate=round(train_wr, 4),
                    test_win_rate=round(test_wr, 4),
                    degradation=round(degradation, 4),
                )
            )

            start += step

        return windows

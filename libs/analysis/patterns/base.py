"""
Abstract base for all candle pattern detectors.

Every detector:
  - Works on an enriched OHLCV DataFrame
  - Examines the TAIL (no full-scan)
  - Returns a PatternResult with graded confidence
  - Never raises on "no pattern found" — returns detected=False
  - Is independently testable

Matching modes:
  - STRICT:  textbook-perfect proportions required
  - BALANCED: moderate tolerance (default)
  - LOOSE:   relaxed matching for noisy/crypto data
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar

import pandas as pd

from libs.core.models.domain import PatternBias, PatternResult


class MatchingMode(str, Enum):
    STRICT = "strict"
    BALANCED = "balanced"
    LOOSE = "loose"


# Confidence modifiers per mode
_MODE_MODIFIERS: dict[MatchingMode, float] = {
    MatchingMode.STRICT: 1.0,
    MatchingMode.BALANCED: 0.9,
    MatchingMode.LOOSE: 0.75,
}

# Threshold multipliers per mode (how lenient proportions are)
_MODE_THRESHOLDS: dict[MatchingMode, float] = {
    MatchingMode.STRICT: 1.0,
    MatchingMode.BALANCED: 0.8,
    MatchingMode.LOOSE: 0.6,
}


class BasePatternDetector(ABC):
    """All candle pattern detectors inherit from this."""

    min_bars_required: ClassVar[int] = 1

    def __init__(self, mode: MatchingMode = MatchingMode.BALANCED) -> None:
        self.mode = mode

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def bias(self) -> PatternBias: ...

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> PatternResult: ...

    def _no_pattern(self, details: dict | None = None) -> PatternResult:
        return PatternResult(
            pattern_name=self.name,
            detected=False,
            confidence=0.0,
            bias=self.bias,
            candle_span=self.min_bars_required,
            details=details or {},
        )

    def _result(
        self,
        confidence: float,
        details: dict | None = None,
        bias: PatternBias | None = None,
        explanation: str = "",
        category: str = "",
        reliability: float = 0.0,
    ) -> PatternResult:
        """Produce a detected=True result with mode-adjusted confidence."""
        adj_conf = min(1.0, confidence * _MODE_MODIFIERS[self.mode])
        return PatternResult(
            pattern_name=self.name,
            detected=True,
            confidence=round(adj_conf, 4),
            bias=bias or self.bias,
            candle_span=self.min_bars_required,
            details=details or {},
            explanation=explanation,
            category=category,
            reliability=reliability,
        )

    def _threshold(self, base: float) -> float:
        """Scale a proportion threshold by the matching mode."""
        return base * _MODE_THRESHOLDS[self.mode]

    @staticmethod
    def _vol_bonus(relative_volume: float) -> float:
        """Small confidence bonus for above-average volume."""
        if relative_volume > 1.5:
            return min(0.10, (relative_volume - 1.5) * 0.05)
        return 0.0

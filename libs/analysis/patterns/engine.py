"""
CandlePatternEngine — runs all 40 pattern detectors against a candle sequence.

Pattern detection is a confluence input ONLY. This engine never produces
BUY or SELL signals. Results feed the ConfluenceEngine for scoring.

Usage:
    engine = CandlePatternEngine()
    patterns = engine.detect(candles)          # detected=True, sorted by confidence desc
    patterns = engine.detect(candles, min_confidence=0.60)  # actionable only
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from libs.analysis.patterns.base import BasePatternDetector, MatchingMode
from libs.analysis.patterns.context_candle import (
    InsideBarDetector,
    OutsideBarDetector,
    PinBarDetector,
    RejectionCandleDetector,
    BreakoutCandleDetector,
    ExhaustionCandleDetector,
    MomentumCandleDetector,
    LongWickCandleDetector,
    NarrowRangeCandleDetector,
    WideRangeCandleDetector,
    TrapCandleDetector,
    FailedBreakoutCandleDetector,
)
from libs.analysis.patterns.multi_candle import (
    MorningStarDetector,
    EveningStarDetector,
    MorningDojiStarDetector,
    EveningDojiStarDetector,
    ThreeWhiteSoldiersDetector,
    ThreeBlackCrowsDetector,
    BullishAbandonedBabyDetector,
    BearishAbandonedBabyDetector,
    RisingThreeMethodsDetector,
    FallingThreeMethodsDetector,
)
from libs.analysis.patterns.single_candle import (
    HammerDetector,
    InvertedHammerDetector,
    ShootingStarDetector,
    HangingManDetector,
    DojiDetector,
    DragonflyDojiDetector,
    GravestoneDojiDetector,
    SpinningTopDetector,
    BullishMarubozuDetector,
    BearishMarubozuDetector,
)
from libs.analysis.patterns.two_candle import (
    BullishEngulfingDetector,
    BearishEngulfingDetector,
    BullishHaramiDetector,
    BearishHaramiDetector,
    PiercingLineDetector,
    DarkCloudCoverDetector,
    TweezerBottomDetector,
    TweezerTopDetector,
)
from libs.core.models.domain import Candle, PatternResult
from libs.core.logging.logger import get_logger

log = get_logger(__name__)


class CandlePatternEngine:
    """
    Runs all 40 pattern detectors against a candle sequence.

    Usage:
        engine = CandlePatternEngine()
        patterns = engine.detect(candles)
        # patterns is list[PatternResult] — detected=True only

    Pattern detection contributes to confluence scoring only.
    The engine NEVER produces BUY/SELL signals.
    """

    def __init__(self, mode: MatchingMode = MatchingMode.BALANCED) -> None:
        self._detectors: list[BasePatternDetector] = [
            # Single-candle (10)
            HammerDetector(mode),
            InvertedHammerDetector(mode),
            ShootingStarDetector(mode),
            HangingManDetector(mode),
            DojiDetector(mode),
            DragonflyDojiDetector(mode),
            GravestoneDojiDetector(mode),
            SpinningTopDetector(mode),
            BullishMarubozuDetector(mode),
            BearishMarubozuDetector(mode),
            # Two-candle (8)
            BullishEngulfingDetector(mode),
            BearishEngulfingDetector(mode),
            BullishHaramiDetector(mode),
            BearishHaramiDetector(mode),
            PiercingLineDetector(mode),
            DarkCloudCoverDetector(mode),
            TweezerBottomDetector(mode),
            TweezerTopDetector(mode),
            # Multi-candle (10)
            MorningStarDetector(mode),
            EveningStarDetector(mode),
            MorningDojiStarDetector(mode),
            EveningDojiStarDetector(mode),
            ThreeWhiteSoldiersDetector(mode),
            ThreeBlackCrowsDetector(mode),
            BullishAbandonedBabyDetector(mode),
            BearishAbandonedBabyDetector(mode),
            RisingThreeMethodsDetector(mode),
            FallingThreeMethodsDetector(mode),
            # Context-candle (12)
            InsideBarDetector(mode),
            OutsideBarDetector(mode),
            PinBarDetector(mode),
            RejectionCandleDetector(mode),
            BreakoutCandleDetector(mode),
            ExhaustionCandleDetector(mode),
            MomentumCandleDetector(mode),
            LongWickCandleDetector(mode),
            NarrowRangeCandleDetector(mode),
            WideRangeCandleDetector(mode),
            TrapCandleDetector(mode),
            FailedBreakoutCandleDetector(mode),
        ]

    @property
    def detector_names(self) -> list[str]:
        """Names of all registered detectors."""
        return [d.name for d in self._detectors]

    def detect(
        self,
        candles: Sequence[Candle],
        min_confidence: float = 0.0,
    ) -> list[PatternResult]:
        """
        Run all detectors against the candle sequence.

        Args:
            candles: Sequence of Candle objects (oldest first).
            min_confidence: Minimum confidence threshold for returned results.

        Returns:
            List of PatternResult with detected=True, sorted by confidence descending.
            candle_index in each result is the 0-based index within the provided candles sequence (not a global bar index).
        """
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0.0, 1.0], got {min_confidence!r}")

        if len(candles) == 0:
            return []

        # Build enriched DataFrame
        records = []
        for c in candles:
            o, h, l, cl, v = c.open, c.high, c.low, c.close, c.volume
            body = abs(cl - o)
            rng = h - l or 1e-9
            records.append({
                "open": o,
                "high": h,
                "low": l,
                "close": cl,
                "volume": v,
                "body_size": body,
                "total_range": rng,
                "body_pct": body / rng,
                "upper_wick": h - max(o, cl),
                "lower_wick": min(o, cl) - l,
                "relative_volume": 1.0,  # no rolling average available without history; neutral
            })
        df = pd.DataFrame(records)

        candle_index = len(candles) - 1
        source_timestamp = candles[-1].timestamp

        detected: list[PatternResult] = []
        for detector in self._detectors:
            if len(df) < detector.min_bars_required:
                continue
            try:
                result = detector.detect(df)
            except Exception as exc:
                log.warning("pattern_detect_failed", detector=detector.name, error=str(exc))
                continue

            if not result.detected:
                continue

            # Stamp the candle index and source timestamp (PatternResult is frozen)
            result = result.model_copy(update={
                "candle_index": candle_index,
                "source_timestamp": source_timestamp,
            })

            if result.confidence >= min_confidence:
                detected.append(result)

        return sorted(detected, key=lambda r: r.confidence, reverse=True)
